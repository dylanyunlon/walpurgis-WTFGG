#!/bin/bash
# run_reverie.sh — Reverie变体实验入口
# Usage: bash run_reverie.sh [DATASET] [DEVICE] [EPOCHS]
#   DATASET: SYNTH | METR-LA | PEMS-BAY | PEMS04 | PEMS08
#   DEVICE:  cpu | cuda:0 | cuda:1 ...
#   EPOCHS:  int (optional override)

set -e
DATASET=${1:-SYNTH}
DEVICE=${2:-cpu}
EPOCHS=${3:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Generate SYNTH data if needed
if [ "$DATASET" = "SYNTH" ]; then
    if [ ! -f "src/walpurgis_reverie/datasets/SYNTH/train.npz" ]; then
        echo "[RV] Generating SYNTH data..."
        python -c "
import sys; sys.path.insert(0,'src')
from walpurgis_reverie.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
    fi
fi

# Run training
ARGS="--dataset $DATASET --device $DEVICE"
if [ -n "$EPOCHS" ]; then
    ARGS="$ARGS --epochs $EPOCHS"
fi

echo "[RV] Starting: python train_reverie.py $ARGS"
python train_reverie.py $ARGS

echo "[RV] Done."
