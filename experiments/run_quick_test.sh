#!/usr/bin/env bash
# experiments/run_quick_test.sh — 先验证pipeline再跑正式实验
# 用法: GPU=2 bash experiments/run_quick_test.sh
# 注意: 前台运行，不要nohup！
set -eo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

GPU="${GPU:-2}"

# Conda
set +e
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate walking3 2>/dev/null
set -e

echo "============================================"
echo " Step 1: SYNTH 3-epoch pipeline validation"
echo "============================================"

if [ ! -f "datasets/SYNTH/train.npz" ]; then
    echo "ERROR: datasets/SYNTH/train.npz not found"
    echo "Generate with: python3 src/walpurgis/generate_synth_data.py"
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" python3 train_walpurgis.py \
    --dataset SYNTH --device cuda:0 --epochs 3 --debug 2>&1 | tail -30

if [ $? -ne 0 ]; then
    echo "!! SYNTH validation FAILED. Fix before running METR-LA."
    exit 1
fi

echo ""
echo "============================================"
echo " SYNTH OK. Step 2: METR-LA 200-epoch run"
echo "============================================"

if [ ! -f "datasets/METR-LA/train.npz" ]; then
    echo "ERROR: datasets/METR-LA/train.npz not found"
    exit 1
fi

export GIT_TOKEN="${GIT_TOKEN:-}"
GPU="$GPU" EPOCHS=200 bash experiments/run_server_experiment.sh

echo "Done."
