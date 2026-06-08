#!/bin/bash
# run_corona.sh — Corona(日冕)变体训练脚本 (conda+gpu环境)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-SYNTH}"
DEVICE="${2:-cpu}"
CONDA_ENV="${CONDA_ENV:-walpurgis}"

# 环境检测: 如果有conda就激活, 没有就用系统python
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda env list | grep -q "^${CONDA_ENV} "; then
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
fi

# GPU检测
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "[corona] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    DEVICE="${DEVICE:-cuda:0}"
fi

echo "=========================================="
echo " Walpurgis Corona — Training Pipeline"
echo " Dataset: $DATASET"
echo " Device:  $DEVICE"
echo "=========================================="

# 依赖安装 (if needed)
pip install pyyaml setproctitle scipy --quiet 2>/dev/null || true

# 生成SYNTH数据
if [ "$DATASET" = "SYNTH" ] && [ ! -f "datasets/SYNTH/train.npz" ]; then
    echo "[corona] Generating SYNTH dataset..."
    python -c "
import sys; sys.path.insert(0, 'src')
from walpurgis_corona.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
fi

# 训练 (不是训练模型! 只是运行实验3个epoch拿数据)
python train_corona.py --dataset "$DATASET" --device "$DEVICE" --epochs 3 --debug
