#!/usr/bin/env bash
# run_eclipse.sh — Eclipse variant training pipeline
set -euo pipefail

DATASET="${DATASET:-SYNTH}"
EPOCHS="${EPOCHS:-3}"
DEVICE="${DEVICE:-cpu}"

export ECLIPSE_DEBUG="${ECLIPSE_DEBUG:-0}"
export EPOCHS

echo "=== D2STGNN Eclipse Variant ==="
echo "Dataset: $DATASET"
echo "Epochs:  $EPOCHS"
echo "Device:  $DEVICE"
echo "Debug:   $ECLIPSE_DEBUG"
echo ""

cd "$(dirname "$0")"

# Generate synth data if needed
if [ "$DATASET" = "SYNTH" ]; then
    if [ ! -f datasets/SYNTH/train.npz ]; then
        echo "[ECL] Generating synthetic data..."
        PYTHONPATH=src python3 -c "
from walpurgis_eclipse.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
        echo ""
    fi
fi

# Train
python3 train_eclipse.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --epochs "$EPOCHS"

echo ""
echo "=== 训练完毕 ==="
echo "模型保存于: output/D2STGNN_${DATASET}.pt"
