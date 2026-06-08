#!/usr/bin/env bash
# run_cathexis.sh — Cathexis(精神贯注) 训练脚本
set -euo pipefail

DATASET="${1:-SYNTH}"
DEVICE="${2:-cpu}"
EPOCHS="${3:-}"

cd "$(dirname "$0")"

echo "============================================"
echo " Cathexis (精神贯注) D2STGNN"
echo " Dataset: ${DATASET}"
echo " Device:  ${DEVICE}"
echo "============================================"

# 如果SYNTH数据不存在，先生成
if [ "$DATASET" = "SYNTH" ] && [ ! -f "datasets/SYNTH/train.npz" ]; then
    echo "[cathexis] Generating SYNTH data..."
    python3 src/walpurgis_cathexis/generate_synth_data.py
fi

ARGS="--dataset ${DATASET} --device ${DEVICE}"
[ -n "${EPOCHS}" ] && ARGS="${ARGS} --epochs ${EPOCHS}"
[ "${DEBUG:-0}" = "1" ] && ARGS="${ARGS} --debug"

python3 train_cathexis.py ${ARGS}
