#!/usr/bin/env bash
# run_tempest.sh — Tempest variant training pipeline
set -euo pipefail

DATASET="${DATASET:-SYNTH}"
EPOCHS="${EPOCHS:-3}"
DEVICE="${DEVICE:-cpu}"

export TEMPEST_DEBUG="${TEMPEST_DEBUG:-0}"
export EPOCHS

echo "=== D2STGNN Tempest Variant ==="
echo "Dataset: $DATASET"
echo "Epochs:  $EPOCHS"
echo "Device:  $DEVICE"
echo "Debug:   $TEMPEST_DEBUG"
echo ""

cd "$(dirname "$0")"

# Generate synth data if needed (fBm + Gabriel graph)
if [ "$DATASET" = "SYNTH" ]; then
    if [ ! -f datasets/SYNTH/train.npz ]; then
        echo "[TEM] Generating synthetic data (fBm + Gabriel graph)..."
        PYTHONPATH=src python3 -c "
from walpurgis_tempest.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
        echo ""
    fi
fi

# Train
python3 train_tempest.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --epochs "$EPOCHS"

echo ""
echo "=== 训练完毕 ==="
echo "模型保存于: output/D2STGNN_${DATASET}.pt"
