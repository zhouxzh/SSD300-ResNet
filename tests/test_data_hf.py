import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

# 仅引入数据加载函数
from ssd300.data_hf import download_and_load_coco, get_train_loader, get_val_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Visualize COCO Ground Truth directly from HF Dataset")
    
    # 默认参数设置
    parser.add_argument("--batch-size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=65, help="总训练轮数")
    parser.add_argument("--lr", type=float, default=2.6e-3, help="基础学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--multistep", nargs='+', type=int, default=[43, 54], help="学习率下降的epoch节点")
    
    # 根据是否有 GPU 自动设置 device
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device_default, help="训练设备 (cuda/cpu)")
    
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

def visualize_tensor_boxes(img_tensor, boxes, labels, category_names, save_path, box_coder=None, is_encoded=False):
    """
    可视化 Tensor 格式的数据
    """
    # 1. Denormalize Image (反归一化)
    # Mean/Std 必须与 transforms 中一致
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    
    img = img_tensor.clone()
    img.mul_(std).add_(mean)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # 2. Handle Boxes
    if is_encoded and box_coder is not None:
        # 解码 SSD Offsets -> Coordinates
        # locs: boxes input [8732, 4], labels: labels input [8732]
        # 只保留正样本
        mask = labels > 0
        if mask.any():
            pos_locs = boxes[mask]
            pos_labels = labels[mask]
            
            # Box decoding logic
            if box_coder.dboxes_xywh.device != img_tensor.device:
                box_coder.dboxes_xywh = box_coder.dboxes_xywh.to(img_tensor.device)
            dboxes = box_coder.dboxes_xywh[mask]
            
            v0, v1 = box_coder.variances
            # cx = loc_cx * v0 * d_w + d_cx
            gx = pos_locs[:, 0] * v0 * dboxes[:, 2] + dboxes[:, 0]
            gy = pos_locs[:, 1] * v0 * dboxes[:, 3] + dboxes[:, 1]
            gw = torch.exp(pos_locs[:, 2] * v1) * dboxes[:, 2]
            gh = torch.exp(pos_locs[:, 3] * v1) * dboxes[:, 3]
            
            x1 = gx - gw/2
            y1 = gy - gh/2
            x2 = gx + gw/2
            y2 = gy + gh/2
            
            decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            # Scale back to 300
            boxes_to_draw = decoded_boxes * box_coder.img_size
            labels_to_draw = pos_labels
        else:
            boxes_to_draw = []
            labels_to_draw = []
    else:
        # 已经是坐标了 (Val loader)
        boxes_to_draw = boxes
        labels_to_draw = labels

    # 3. Plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    
    if len(boxes_to_draw) > 0:
        if isinstance(boxes_to_draw, torch.Tensor):
            boxes_to_draw = boxes_to_draw.cpu().numpy()
        if isinstance(labels_to_draw, torch.Tensor):
            labels_to_draw = labels_to_draw.cpu().numpy()
            
        for bbox, cat_id in zip(boxes_to_draw, labels_to_draw):
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            
            # 获取类别名称
            display_txt = str(int(cat_id))
            if category_names and 0 <= cat_id < len(category_names):
                cat_name = category_names[int(cat_id)]
                display_txt = f"{int(cat_id)}: {cat_name}"

            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(xmin, ymin, display_txt, color='white', fontsize=10, backgroundcolor='red', 
                    bbox=dict(facecolor='red', alpha=0.5, pad=0))
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    args = get_args()

    # 1. 加载数据
    print("Loading data...")
    full_dataset = download_and_load_coco()
    if full_dataset is None:
        print("Failed to load dataset.")
        exit(1)
    
    # 获取类别名称列表
    category_names = get_category_names(full_dataset)
    if category_names:
        print(f"Found {len(category_names)} category names.")
    else:        
        print("No category names found. Will display category IDs only.")
        
    save_dir = "debug_gt_viz"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    train_loader = get_train_loader(full_dataset, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    val_loader = get_val_dataloader(full_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 2. 可视化训练集前10张图片的 GT 框 (Decoder Needed)
    print("Visualizing GT boxes for training set...")
    box_coder = train_loader.dataset.box_coder
    
    for i, batch in enumerate(train_loader):
        if i >= 1: break # Visualize just the first batch
        
        # Train Loader returns: images, encoded_locs, encoded_labels
        images, locs, labels = batch
        
        for j in range(min(10, images.size(0))):
            img_t = images[j]
            loc_t = locs[j]
            lbl_t = labels[j]
            
            save_path = f"debug_gt_viz/train_{i}_{j}.jpg"
            visualize_tensor_boxes(img_t, loc_t, lbl_t, category_names, save_path, box_coder, is_encoded=True)
            print(f"Saved {save_path}")

    print("Visualization train loader completed.")
    
    # 3. 可视化验证集
    print("Visualizing GT boxes for val set...")
    for i, batch in enumerate(val_loader):
        if i >= 1: break
        
        # Val Loader returns: images, boxes(list), labels(list), img_ids
        # 这里 colate_fn 不同，boxes 是 list of tensors
        images, boxes_list, labels_list, img_ids = batch
        
        for j in range(min(10, images.size(0))):
            img_t = images[j]
            gt_boxes = boxes_list[j]
            gt_lbls = labels_list[j]
            
            save_path = f"debug_gt_viz/val_{i}_{j}.jpg"
            visualize_tensor_boxes(img_t, gt_boxes, gt_lbls, category_names, save_path, is_encoded=False)
            print(f"Saved {save_path}")

    print("Visualization val completed. Check the 'debug_gt_viz' directory for results.")

