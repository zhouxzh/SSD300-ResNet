import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


# 引入 pycocotools 用于计算 COCO mAP
from pycocotools.cocoeval import COCOeval

# 引入项目中的模型和工具
from ssd.model import SSD300, ResNet, Loss
from ssd.utils import dboxes300_coco, Encoder, visualize_sample


WEIGHTS_DIR = "weights"
BEST_CHECKPOINT = os.path.join(WEIGHTS_DIR, "best.pth")
LAST_CHECKPOINT = os.path.join(WEIGHTS_DIR, "last.pth")

def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

def export_onnx_model(model, device, onnx_path):
    print(f"正在导出 ONNX 模型至 {onnx_path}...")
    model.eval()
    dummy_input = torch.randn(1, 3, 300, 300).to(device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            input_names=['input'],
            output_names=['boxes', 'scores'],
            opset_version=11
        )
        print(f"ONNX 模型已导出至: {onnx_path}")
    except Exception as e:
        print(f"导出 ONNX 失败: {e}")

def validate_and_visualize(ssd_model, epoch, val_loader, eval_encoder, coco_gt, category_names, device, writer, args):
    # 每个 Epoch 结束进行验证
    print(f"Epoch {epoch+1} 结束, 开始评估验证集 mAP 并进行可视化...")
    ssd_model.eval()
    
    # 准备可视化目录
    viz_dir = f"viz_results/epoch_{epoch+1}"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    viz_count = 0
    results_coco = []
    val_map = None
     
    with torch.no_grad():
        for i, (v_images, v_boxes_list, v_labels_list, v_img_ids) in enumerate(val_loader):
            v_images = v_images.to(device)
            
            # 前向推理
            locs, confs = ssd_model(v_images)
            
            # 解码预测结果
            results = eval_encoder.decode_batch(locs, confs)
            
            # --- 可视化逻辑 (前10张) ---
            if viz_count < 10:
                for b in range(len(v_images)):
                    if viz_count >= 10: break
                    
                    p_boxes, p_labels, p_scores = results[b]
                    
                    visualize_sample(
                        v_images[b],
                        v_boxes_list[b],
                        v_labels_list[b],
                        p_boxes,
                        p_labels,
                        p_scores,
                        category_names,
                        os.path.join(viz_dir, f"val_{viz_count}.jpg")
                    )
                    viz_count += 1
            # -----------------------------------
            
            for b in range(len(results)):
                p_boxes, p_labels, p_scores = results[b]
                
                if p_boxes.numel() == 0:
                    continue

                p_boxes *= 300.0
                
                img_id = v_img_ids[b]
                if torch.is_tensor(img_id):
                    img_id = img_id.item()

                for i in range(len(p_boxes)):
                    box = p_boxes[i].tolist()
                    results_coco.append({
                        "image_id": img_id,
                        "category_id": p_labels[i].item(),
                        "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                        "score": p_scores[i].item()
                    })
        
        if results_coco:
            print(f"收集到 {len(results_coco)} 条预测结果，正在计算 mAP...")
            coco_dt = coco_gt.loadRes(results_coco)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            val_map = coco_eval.stats[0] # mAP @ IoU=0.50:0.95
            print(f"Epoch [{epoch+1}/{args.epochs}] mAP: {val_map:.4f}")
            writer.add_scalar('Val/mAP', val_map, epoch)
        else:
            print(f"Epoch {epoch+1}: 未检测到任何目标，results_coco 为空。")

    return val_map


def save_checkpoint(path, epoch, ssd_model, optimizer, scheduler, best_map):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": ssd_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map": best_map,
    }, path)


def train(args, train_loader, val_loader, coco_gt, category_names=None, resume_checkpoint=None, start_epoch=0):
    writer = SummaryWriter(log_dir=f"logs/{args.backbone}")
    
    # Hyperparameters from args
    batch_size = args.batch_size
    epochs = args.epochs
    n_gpu = 1 # 假设单卡训练
    
    # Learning Rate Calculation
    lr = args.lr * n_gpu * (batch_size / 32)
    print(f"Batch Size: {batch_size}, Calculated Learning Rate: {lr}")

    # 2. 设置 SSD 相关组件
    dboxes = dboxes300_coco()
    eval_encoder = Encoder(dboxes)
    
    # 3. 初始化模型
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    ssd_model = SSD300(backbone=ResNet(backbone=args.backbone, weights='IMAGENET1K_V1'))
    ssd_model.to(device)
    
    resume_state = None
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        if os.path.exists(resume_checkpoint):
            resume_state = torch.load(resume_checkpoint, map_location=device)
            if isinstance(resume_state, dict) and "model_state_dict" in resume_state:
                ssd_model.load_state_dict(resume_state["model_state_dict"])
                start_epoch = int(resume_state.get("epoch", start_epoch - 1)) + 1
                print("Checkpoint loaded successfully.")
            else:
                ssd_model.load_state_dict(resume_state)
                print("Legacy checkpoint loaded successfully.")
        else:
            print(f"Checkpoint file not found: {resume_checkpoint}")

    ssd_model.train()
    
    criterion = Loss(dboxes).to(device)
    
    # 使用 tencent_trick 优化器配置
    optimizer = optim.SGD(tencent_trick(ssd_model), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if isinstance(resume_state, dict):
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])

    if start_epoch > 0 and not (isinstance(resume_state, dict) and "scheduler_state_dict" in resume_state):
        print(f"Advancing scheduler to epoch {start_epoch}")
        for _ in range(start_epoch):
            scheduler.step()

    scaler = torch.amp.GradScaler('cuda')
    
    # 5. 训练循环
    warmup_iters = 300
    # Calculate initial iteration count based on start_epoch
    iteration = start_epoch * len(train_loader)
    best_map = float(resume_state.get("best_map", float("-inf"))) if isinstance(resume_state, dict) else float("-inf")

    print(f"开始训练 from Epoch {start_epoch} (Iteration {iteration})...")
    
    for epoch in range(start_epoch, epochs):
        ssd_model.train()
        for batch_idx, (images, plocs, plabels) in enumerate(train_loader):
            # Warmup only applies in the very beginning of training (first few iterations global)
            # If resetting, iteration is large, so warmup condition `iteration < warmup_iters` will likely be false, which is correct.
            warmup(optimizer, warmup_iters, iteration, lr)

            images = images.to(device)
            plocs = plocs.to(device)
            plabels = plabels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loc_preds, conf_preds = ssd_model(images)
                loc_preds = loc_preds.float()
                conf_preds = conf_preds.float()
                
                gloc = plocs.transpose(1, 2).contiguous()
                glabel = plabels
                loss = criterion(loc_preds, conf_preds, gloc, glabel)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iteration += 1

            # 记录到 Tensorboard，不再频繁 print
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), iteration)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], iteration)
                        
        scheduler.step()
        
        # 调用封装好的验证与可视化函数
        val_map = validate_and_visualize(ssd_model, epoch, val_loader, eval_encoder, coco_gt, category_names, device, writer, args)

        # 只保留 last.pth 和 best.pth
        if val_map is not None:
            best_map = max(best_map, val_map)

        save_checkpoint(LAST_CHECKPOINT, epoch, ssd_model, optimizer, scheduler, best_map)
        if val_map is not None and val_map > best_map:
            best_map = val_map
            save_checkpoint(BEST_CHECKPOINT, epoch, ssd_model, optimizer, scheduler, best_map)
            print(f"New best checkpoint saved to {BEST_CHECKPOINT} with mAP {best_map:.4f}")

    writer.close()
    return ssd_model
