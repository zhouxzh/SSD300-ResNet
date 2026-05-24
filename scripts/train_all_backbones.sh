#!/usr/bin/env bash

set -euo pipefail

BACKBONES=(
  resnet18
  resnet34
  resnet50
  resnet101
  resnet152
  resnet18d
  resnet34d
  resnet50d
  resnet101d
  resnet152d
  resnext50_32x4d
  resnext101_32x8d
  wide_resnet50_2
  wide_resnet101_2
  resnetrs50
  resnetrs101
  resnetrs152
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/scripts/train.py"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Cannot find training script: $TRAIN_SCRIPT" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${CONDA_PREFIX:-}" && -d "$HOME/anaconda3/envs/torch" && -x "$HOME/anaconda3/envs/torch/bin/python" ]]; then
  PYTHON_BIN="$HOME/anaconda3/envs/torch/bin/python"
fi

echo "Using Python: $PYTHON_BIN"
echo "Extra args: $*"

for backbone in "${BACKBONES[@]}"; do
  echo
  echo "===== Training backbone: $backbone ====="
  "$PYTHON_BIN" "$TRAIN_SCRIPT" --backbone "$backbone" "$@"
done

echo
echo "All backbones finished."
