#!/usr/bin/env bash
# experiments/run_metrla.sh — 在 GPU 服务器上训练 Walpurgis D2STGNN 并评估
# 用法: bash experiments/run_metrla.sh [--gpu 0] [--epochs 100] [--tag baseline]
# 结果自动写入 experiments/results/ 并 push 到 git
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── 参数解析 ────────────────────────────────────────────
GPU="${GPU:-2}"         # 默认用 H100 NVL (GPU2)
EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-walpurgis}"
DATASET="${DATASET:-METR-LA}"
SEED="${SEED:-42}"
PUSH="${PUSH:-1}"       # 是否自动 push

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU="$2";    shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --tag)    TAG="$2";    shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        --seed)   SEED="$2";   shift 2;;
        --no-push) PUSH=0;    shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ID="${TAG}_${DATASET}_${TIMESTAMP}"
RESULT_DIR="experiments/results/${RUN_ID}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Walpurgis Training Run"
echo " ID:      $RUN_ID"
echo " Dataset: $DATASET"
echo " GPU:     $GPU"
echo " Epochs:  $EPOCHS"
echo " Seed:    $SEED"
echo "============================================"

# ── 激活 conda ──────────────────────────────────────────
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walpurgis 2>/dev/null || true
set -u

# ── 检查数据 ────────────────────────────────────────────
if [ ! -f "datasets/${DATASET}/train.npz" ]; then
    echo "ERROR: datasets/${DATASET}/train.npz not found"
    echo "Run: bash experiments/setup_env.sh first"
    exit 1
fi

# ── 训练 ────────────────────────────────────────────────
echo "[TRAIN] Starting at $(date '+%H:%M:%S')..."

CUDA_VISIBLE_DEVICES="$GPU" python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device cuda:0 \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --save_dir "$RESULT_DIR" \
    2>&1 | tee "${RESULT_DIR}/train.log"

echo "[TRAIN] Finished at $(date '+%H:%M:%S')"

# ── 评估 ────────────────────────────────────────────────
echo "[EVAL] Running evaluation..."
CUDA_VISIBLE_DEVICES="$GPU" python3 -c "
import torch, numpy as np, json, sys, os
sys.path.insert(0, 'src')

# 加载最优模型
model_path = '${RESULT_DIR}/best_model.pt'
if not os.path.exists(model_path):
    # fallback: 找到 result_dir 下任意 .pt
    pts = [f for f in os.listdir('${RESULT_DIR}') if f.endswith('.pt')]
    if pts:
        model_path = os.path.join('${RESULT_DIR}', pts[0])
    else:
        print('No model checkpoint found')
        sys.exit(1)

print(f'Model: {model_path}')
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
print(f'Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else \"raw state_dict\"}')

# 结果概要写入 JSON
result = {
    'run_id': '${RUN_ID}',
    'dataset': '${DATASET}',
    'gpu': '${GPU}',
    'epochs': ${EPOCHS},
    'seed': ${SEED},
    'timestamp': '${TIMESTAMP}',
}

# 如果训练产出了 metrics.csv, 解析最终结果
metrics_path = '${RESULT_DIR}/metrics.csv'
if os.path.exists(metrics_path):
    import pandas as pd
    df = pd.read_csv(metrics_path)
    best = df.loc[df['val_mae'].idxmin()]
    result['best_epoch'] = int(best.get('epoch', -1))
    result['best_val_mae'] = float(best['val_mae'])
    result['best_val_rmse'] = float(best.get('val_rmse', 0))
    print(f'Best epoch: {result[\"best_epoch\"]}  val_MAE: {result[\"best_val_mae\"]:.4f}')

with open('${RESULT_DIR}/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Result written to ${RESULT_DIR}/result.json')
" 2>&1 | tee -a "${RESULT_DIR}/eval.log"

# ── Auto-push ───────────────────────────────────────────
if [ "$PUSH" = "1" ]; then
    echo "[GIT] Pushing results..."
    cd "$REPO_DIR"
    git add "experiments/results/${RUN_ID}/"
    git commit -m "experiment: ${RUN_ID} — ${DATASET} ${EPOCHS}ep seed${SEED}" || true
    git push origin main || echo "Push failed (check credentials)"
fi

echo ""
echo "============================================"
echo " Run complete: $RUN_ID"
echo " Results: $RESULT_DIR/"
echo "============================================"
