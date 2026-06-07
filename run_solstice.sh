#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
export SOLSTICE_DEBUG="${SOLSTICE_DEBUG:-1}"
EPOCHS="${EPOCHS:-2}"
DEVICE="${DEVICE:-cpu}"

echo "============================================"
echo "  walpurgis_solstice D2STGNN Training"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "  Debug: $SOLSTICE_DEBUG"
echo "============================================"

# Step 1: Generate synthetic data
echo "[1/2] Generating synthetic data..."
python -m walpurgis_solstice.generate_synth_data

# Step 2: Train
echo "[2/2] Training Solstice model..."
python train_solstice.py --config configs/SYNTH.yaml --device "$DEVICE" --epochs "$EPOCHS"

echo "============================================"
echo "  Solstice training complete!"
echo "============================================"
