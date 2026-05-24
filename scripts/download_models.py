import os
from huggingface_hub import hf_hub_download
import argparse
import shutil

from _bootstrap import add_root_path

add_root_path()

from ssd300.model import AVAILABLE_RESNET_BACKBONES
from ssd300.train import get_onnx_path


def download_onnx_model(backbone):
    repo_id = "zhouxzh/ssd"
    local_target = get_onnx_path(backbone)
    remote_filenames = [f"ssd300_{backbone}.onnx", f"ssd_{backbone}.onnx"]

    # 如果目标文件已存在，直接返回路径（避免重复下载）
    if os.path.exists(local_target):
        print(f"模型已存在：{local_target}")
        return local_target

    # 确保目标目录存在
    os.makedirs(os.path.dirname(local_target), exist_ok=True)

    last_error = None
    for filename in remote_filenames:
        try:
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"缓存路径：{cached_path}")

            # 统一保存为 ssd300_ 前缀
            shutil.copy2(cached_path, local_target)
            print(f"模型已复制到：{local_target}")
            return local_target
        except Exception as e:
            last_error = e

    raise last_error
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", help="backbone name")
    args = parser.parse_args()
    if args.backbone == "all":
        for backbone in AVAILABLE_RESNET_BACKBONES:
            try:
                download_onnx_model(backbone)
            except Exception as e:
                print(f"下载 {backbone} 失败: {e}")
    else:
        download_onnx_model(args.backbone)
