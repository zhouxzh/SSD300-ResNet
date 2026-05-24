import os
import argparse
import torch

from _bootstrap import add_root_path

add_root_path()

from ssd300.data_hf import download_and_load_coco, get_train_loader, get_val_dataloader, get_coco_ground_truth
from ssd300.model import AVAILABLE_RESNET_BACKBONES
from ssd300.train import train
from ssd300.train import export_onnx_model, get_onnx_path
from pycocotools.coco import COCO
    
def get_args():
    parser = argparse.ArgumentParser(description="Visualize COCO Ground Truth directly from HF Dataset")
    
    # Default settings
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=65, help="Total epochs")
    parser.add_argument("--lr", type=float, default=2.6e-3, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--multistep", nargs='+', type=int, default=[43, 54], help="Epochs to decay learning rate")
    
    # Device setting
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device_default, help="Device (cuda/cpu)")
    
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=AVAILABLE_RESNET_BACKBONES + ["all"],
        help="Model backbone, or 'all' for all supported resnet variants",
    )
    parser.add_argument("--augment", action='store_true', default=True, help="Use data augmentation")
    
    # Restart from weights/<backbone>/last.pth
    parser.add_argument("--restart", action='store_true', help="Resume training from weights/<backbone>/last.pth")
    
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

def find_latest_checkpoint(backbone):
    """
    Resume from weights/<backbone>/last.pth if available.
    The checkpoint stores the completed epoch count in its metadata.
    """
    model_dir = os.path.join("weights", backbone)
    last_path = os.path.join(model_dir, "last.pth")
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu")
        if isinstance(state, dict) and "epoch" in state:
            print(f"Found last checkpoint: {last_path} (Epoch {state['epoch']})")
            return last_path, int(state["epoch"])
        print(f"Found legacy last checkpoint: {last_path}")
        return last_path, 0

    best_path = os.path.join(model_dir, "best.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        if isinstance(state, dict) and "epoch" in state:
            print(f"Found best checkpoint: {best_path} (Epoch {state['epoch']})")
            return best_path, int(state["epoch"])
        print(f"Found legacy best checkpoint: {best_path}")
        return best_path, 0

    print(f"No checkpoint found for backbone {backbone} in weights/.")
    return None, 0

if __name__ == "__main__":
    # 1. Get arguments
    args = get_args()

    # 2. Load Data
    print("Loading training data...")
    full_dataset = download_and_load_coco()
    train_loader = get_train_loader(full_dataset, args.batch_size, num_workers=args.num_workers, args=args)
    
    print("Loading validation data...")
    val_loader = get_val_dataloader(full_dataset, args.batch_size, num_workers=args.num_workers)
    
    # 3. Prepare COCO Ground Truth
    gt_file = get_coco_ground_truth(full_dataset['val'])
    coco_gt = COCO(gt_file)
    # Get category names
    category_names = get_category_names(full_dataset)
    if category_names:
        category_names = ['BACKGROUND'] + category_names
    
    # 4. Handle Restart/Resume and training mode
    backbones = AVAILABLE_RESNET_BACKBONES if args.backbone == "all" else [args.backbone]

    for backbone_name in backbones:
        args.backbone = backbone_name
        resume_checkpoint = None
        start_epoch = 0
        if args.restart:
            resume_checkpoint, start_epoch = find_latest_checkpoint(backbone_name)

        print(f"\n===== Training backbone: {backbone_name} =====")
        ssd_model = train(args, train_loader, val_loader, coco_gt, category_names, resume_checkpoint, start_epoch)
        export_onnx_model(ssd_model, args.device, onnx_path=get_onnx_path(backbone_name))
