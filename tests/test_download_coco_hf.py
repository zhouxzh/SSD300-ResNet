import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 仅引入数据加载函数
from ssd300.data_hf import download_and_load_coco

def get_args():
    parser = argparse.ArgumentParser(description="Visualize COCO Ground Truth directly from HF Dataset")
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

def visualize_gt_boxes(item, category_names, save_path="gt_viz.png"):
    # 原始图片 (PIL)
    image = item['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 原始标注 (HuggingFace 格式: x, y, w, h，绝对坐标)
    objects = item['objects']
    # image_id can be missing or int
    img_id = item.get('image_id', i)
    
    # 绘图 Setup
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    
    bboxes = objects['bbox'] 
    categories = objects['category']
    
    # 遍历该图片的所有框
    for bbox, cat_id in zip(bboxes, categories):
        # 修改: 按照 [xmin, ymin, xmax, ymax] 解析
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        
        print(f"Image {img_id} - Image size:{image.size} - Box: ({xmin}, {ymin}, {xmax}, {ymax}), Category ID: {cat_id}")
        
        # 获取类别名称
        display_txt = str(cat_id)
        if category_names and 0 <= cat_id < len(category_names):
            cat_name = category_names[cat_id]
            display_txt = f"{cat_id}: {cat_name}"

        # 画矩形 (Matplotlib Rectangle 接受 xy, width, height)
        # 注意: xy 参数是左上角坐标 (xmin, ymin)
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 画类别标签 (名字 + ID)
        ax.text(xmin, ymin, display_txt, color='white', fontsize=10, backgroundcolor='red', 
                bbox=dict(facecolor='red', alpha=0.5, pad=0))
    
    # 保存图片
    plt.axis('off')
    save_path = os.path.join(save_dir, f"{split}_{i}_id{img_id}.jpg")
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
        
    splits = ['train', 'val']
    
    print(f"Visualizing Ground Truth boxes to folder: {save_dir}")

    for split in splits:
        if split not in full_dataset:
            print(f"Split {split} not found in dataset.")
            continue
            
        print(f"Processing split: {split}")
        dataset_split = full_dataset[split]
        
        # 只处理前10个样本
        num_to_viz = min(len(dataset_split), 10)
        
        for i in range(num_to_viz):
            item = dataset_split[i]
            save_path = os.path.join(save_dir, f"{split}_sample_{i}.png")
            visualize_gt_boxes(item, category_names, save_path=save_path)          
    print("Done. Please check the output directory.")
