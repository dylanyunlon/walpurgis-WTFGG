#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
export EQUINOX_DEBUG="${EQUINOX_DEBUG:-1}"
EPOCHS="${EPOCHS:-2}"
DEVICE="${DEVICE:-cpu}"

echo "============================================"
echo "  walpurgis_equinox D2STGNN Training"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "  Debug: $EQUINOX_DEBUG"
echo "  Algorithms: LogCosh | Linformer | WeightNorm"
echo "  Lookahead(Adam)+OneCycleLR | Highway GRU"
echo "  DenseNet connections | Gumbel-Softmax | CutMix"
echo "============================================"

# Step 1: Generate synthetic data
echo "[1/2] Generating synthetic data..."
python -m walpurgis_equinox.generate_synth_data

# Step 2: Train
echo "[2/2] Training Equinox model..."
python train_equinox.py --config configs/SYNTH.yaml --device "$DEVICE" --epochs "$EPOCHS"

echo "============================================"
echo "  Equinox training complete!"
echo "============================================"
