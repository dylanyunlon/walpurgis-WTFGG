#!/bin/bash
# ============================================================
# CardGame Training Pipeline (D2STGNN CardGame variant)
# 从项目根目录运行: bash run_cardgame.sh
#
# 环境变量:
#   DATASET          数据集 (默认: SYNTH)
#   DEVICE           设备  (默认: cpu)
#   EPOCHS           轮数  (默认: 使用config值)
#   CARDGAME_DEBUG   调试  (默认: 0)
# ============================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-SYNTH}"
DEVICE="${DEVICE:-cpu}"
EPOCHS_ARG="${EPOCHS:+--epochs $EPOCHS}"
DEBUG="${CARDGAME_DEBUG:-0}"

echo "=========================================="
echo "   CardGame (D2STGNN) Training Pipeline"
echo "   Dataset : $DATASET"
echo "   Device  : $DEVICE"
echo "   Debug   : $DEBUG"
echo "=========================================="
echo ""

# ── Step 0: 环境检查 ────────────────────────────────────────
echo "=== Step 0: 环境检查 ==="
python3 -c "
import torch, numpy, yaml, sklearn
print(f'  Python  : ok')
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
print(f'  NumPy   : {numpy.__version__}')
"
echo ""

# ── Step 1: 生成合成数据 (仅SYNTH数据集需要) ──────────────────
if [ "$DATASET" = "SYNTH" ]; then
    echo "=== Step 1: 生成合成数据 ==="
    if [ ! -f "datasets/SYNTH/train.npz" ]; then
        PYTHONPATH=src python3 -c "
from walpurgis_cardgame.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
        echo "  ✓ 合成数据生成完毕"
    else
        echo "  ✓ 合成数据已存在, 跳过生成"
    fi
    echo ""
fi

# ── Step 2: 训练 ────────────────────────────────────────────
echo "=== Step 2: 训练 ==="
export CARDGAME_DEBUG="$DEBUG"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="--debug"
fi

python3 train_cardgame.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    $EPOCHS_ARG \
    $DEBUG_FLAG

echo ""
echo "=== 训练完毕 ==="
echo "模型保存于: output/D2STGNN_${DATASET}.pt"
