import os
import torch
import torch.optim as optim
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from _bootstrap import add_root_path

add_root_path()

# 引入 pycocotools 用于计算 COCO mAP
from pycocotools.cocoeval import COCOeval

# 引入项目中的模型和工具
from ssd300.model import SSD300, ResNet, Loss
from ssd300.utils import dboxes300_coco, Encoder

# 引入数据处理模块
from ssd300.data_hf import download_and_load_coco, get_train_loader, get_val_dataloader, get_coco_ground_truth

def get_args():
    parser = argparse.ArgumentParser(description="Visualize COCO Ground Truth directly from HF Dataset")
    
    # 默认参数设置
    parser.add_argument("--batch-size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="总训练轮数")
    parser.add_argument("--lr", type=float, default=2.6e-3, help="基础学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--multistep", nargs='+', type=int, default=[43, 54], help="学习率下降的epoch节点")
    
    # 根据是否有 GPU 自动设置 device
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device_default, help="训练设备 (cuda/cpu)")
    
    parser.add_argument("--backbone", type=str, default="resnet50", help="Model backbone")
    parser.add_argument("--augment", action='store_true', default=True, help="是否使用数据增强")
    
    return parser.parse_args()

def get_category_names(dataset):
    try:
        features = dataset['train'].features
        if 'objects' in features:
            objects_feat = features['objects']
            
            def safe_get(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                if hasattr(obj, key):
                    return getattr(obj, key)
                if key == 'feature' and hasattr(obj, 'feature'):
                    return obj.feature
                return None

            category_feat = safe_get(objects_feat, 'category')
            if category_feat is None:
                inner_feat = safe_get(objects_feat, 'feature')
                if inner_feat:
                    category_feat = safe_get(inner_feat, 'category')

            if category_feat:
                cat_inner = safe_get(category_feat, 'feature')
                target_feat = cat_inner if cat_inner is not None else category_feat
                names = safe_get(target_feat, 'names')
                if names and isinstance(names, list):
                    return names

        print("Warning: Could not find category names in dataset features.")
        return None

    except Exception as e:
        print(f"Error extracting category names: {e}")
        return None

def visualize_validation_sample(img_tensor, gt_boxes, gt_labels, p_boxes, p_labels, p_scores, category_names, save_path):
    """
    可视化单张样本的 GT 和 预测结果，替代原有的 visualize_tensor_boxes
    """
    # 1. Denormalize Image
    device = img_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    
    img = img_tensor.clone()
    img.mul_(std).add_(mean)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    # 2. 画 Ground Truth (绿色)
    if isinstance(gt_boxes, torch.Tensor): 
        gt_boxes = gt_boxes.cpu().numpy()
        
    for box, lbl in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        cat_n = str(lbl.item())
        if category_names and lbl.item() < len(category_names):
            cat_n = category_names[lbl.item()]
        ax.text(xmin, ymin, f"GT: {cat_n}", color='lime', fontsize=9, backgroundcolor='black', alpha=0.6)
        
    # 3. 画 Prediction (红色)
    # p_boxes 是 normalized [0,1], 需要 scale 到 300
    if p_boxes.numel() > 0:
        p_boxes_np = p_boxes.cpu().numpy() * 300.0 
        p_labels_np = p_labels.cpu().numpy()
        p_scores_np = p_scores.cpu().numpy()
        
        for box, lbl, scr in zip(p_boxes_np, p_labels_np, p_scores_np):
            # 简单的可视化阈值
            if scr < 0.4: continue
            
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            cat_n = str(lbl)
            if category_names and lbl < len(category_names):
                cat_n = category_names[lbl]
            ax.text(xmin, ymax, f"Pred: {cat_n} {scr:.2f}", color='white', fontsize=9, backgroundcolor='red', alpha=0.7)
            
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

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
    
    # 反归一化参数移动到了 visualize_validation_sample 函数内
    
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
                    
                    visualize_validation_sample(
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


def train(args, train_loader, val_loader, coco_gt, category_names=None):
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
    ssd_model.train()
    
    criterion = Loss(dboxes).to(device)
    
    # 使用 tencent_trick 优化器配置
    optimizer = optim.SGD(tencent_trick(ssd_model), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. 训练循环
    warmup_iters = 300
    iteration = 0

    print("开始训练...")
    
    for epoch in range(epochs):
        ssd_model.train()
        for batch_idx, (images, plocs, plabels) in enumerate(train_loader):
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
        validate_and_visualize(ssd_model, epoch, val_loader, eval_encoder, coco_gt, category_names, device, writer, args)

        # 保存检查点
        os.makedirs("models", exist_ok=True)
        torch.save(ssd_model.state_dict(), f"models/ssd300_{args.backbone}_{epoch}.pth")

    writer.close()
    return ssd_model


if __name__ == "__main__":
    # 1. 获取参数
    args = get_args()

    # 2. 获取数据 (放到主程序中)
    print("正在加载训练数据...")
    full_dataset = download_and_load_coco()
    train_loader = get_train_loader(full_dataset, args.batch_size, num_workers=args.num_workers, args=args)
    
    print("正在加载验证数据...")
    val_loader = get_val_dataloader(full_dataset, args.batch_size, num_workers=args.num_workers)
    
    # 3. 准备 COCO 真值
    coco_gt = get_coco_ground_truth(full_dataset)
    # 获取类别名称
    category_names = get_category_names(full_dataset)
    if category_names:
        category_names = ['BACKGROUND'] + category_names
    
    # 4. 启动训练
    ssd_model = train(args, train_loader, val_loader, coco_gt, category_names)
    
    # 训练完成后导出 ONNX 模型
    export_onnx_model(ssd_model, args.device, onnx_path=f"ssd300_{args.backbone}.onnx")
