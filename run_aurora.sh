#!/usr/bin/env bash
# run_aurora.sh — Aurora变体训练启动脚本
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${1:-SYNTH}"
DEVICE="${2:-cpu}"
EPOCHS="${3:-}"
DEBUG="${4:---debug}"

CMD="python train_aurora.py --dataset $DATASET --device $DEVICE"
[ -n "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS"
[ "$DEBUG" = "--debug" ] && CMD="$CMD --debug"

echo "[Aurora] $CMD"
exec $CMD
