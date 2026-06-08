#!/bin/bash
# run_perihelion.sh — Perihelion变体训练启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-SYNTH}"
DEVICE="${2:-cpu}"
DEBUG_FLAG=""

if [ "${3}" = "--debug" ] || [ "${DEBUG:-0}" = "1" ]; then
    DEBUG_FLAG="--debug"
fi

echo "========================================"
echo " Walpurgis Perihelion — Training Pipeline"
echo " Dataset: $DATASET"
echo " Device:  $DEVICE"
echo " Debug:   ${DEBUG_FLAG:-off}"
echo "========================================"

# 如果使用SYNTH数据集，先生成数据
if [ "$DATASET" = "SYNTH" ]; then
    if [ ! -f "datasets/SYNTH/train.npz" ]; then
        echo "[run_perihelion] Generating SYNTH dataset..."
        python -c "
import sys; sys.path.insert(0, 'src')
from walpurgis_perihelion.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
    fi
fi

python train_perihelion.py --dataset "$DATASET" --device "$DEVICE" $DEBUG_FLAG
