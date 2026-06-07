#!/usr/bin/env bash
# run_nebula.sh — Nebula variant training pipeline
set -euo pipefail

DATASET="${DATASET:-SYNTH}"
EPOCHS="${EPOCHS:-3}"
DEVICE="${DEVICE:-cpu}"

export NEBULA_DEBUG="${NEBULA_DEBUG:-0}"
export EPOCHS

echo "=== D2STGNN Nebula Variant ==="
echo "Dataset: $DATASET"
echo "Epochs:  $EPOCHS"
echo "Device:  $DEVICE"
echo "Debug:   $NEBULA_DEBUG"
echo ""

cd "$(dirname "$0")"

# Generate synth data if needed
if [ "$DATASET" = "SYNTH" ]; then
    if [ ! -f datasets/SYNTH/train.npz ]; then
        echo "[NEB] Generating synthetic data..."
        PYTHONPATH=src python3 -c "
from walpurgis_nebula.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
        echo ""
    fi
fi

# Train
python3 train_nebula.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --epochs "$EPOCHS"

echo ""
echo "=== 训练完毕 ==="
echo "模型保存于: output/D2STGNN_${DATASET}.pt"
