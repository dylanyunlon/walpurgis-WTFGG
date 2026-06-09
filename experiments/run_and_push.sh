#!/usr/bin/env bash
# run_and_push.sh — 运行实验 + 收集结构化结果 + 自动push到git
# 用法: DATASET=SYNTH EPOCHS=5 bash experiments/run_and_push.sh
# 服务器: GPU=2 DATASET=METR-LA EPOCHS=200 SEED=42 bash experiments/run_and_push.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── 参数 ──
DATASET="${DATASET:-SYNTH}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
GPU="${GPU:-}"
DEBUG="${DEBUG:-1}"
DEVICE="cpu"
if [ -n "$GPU" ]; then
    DEVICE="cuda:${GPU}"
fi

RUN_ID="${DATASET}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="experiments/results"
LOG_FILE="${SAVE_DIR}/${RUN_ID}.log"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Walpurgis Experiment Runner                            ║"
echo "║  Dataset : ${DATASET}                                   "
echo "║  Device  : ${DEVICE}                                    "
echo "║  Epochs  : ${EPOCHS}                                    "
echo "║  Seed    : ${SEED}                                      "
echo "║  Run ID  : ${RUN_ID}                                    "
echo "╚══════════════════════════════════════════════════════════╝"

mkdir -p "$SAVE_DIR"

# ── 运行训练 ──
WALPURGIS_DEBUG=${DEBUG} python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    ${DEBUG:+--debug} \
    2>&1 | tee "$LOG_FILE"

# ── 提取结果到JSON ──
python3 -c "
import re, json, sys

log_file = '${LOG_FILE}'
with open(log_file) as f:
    content = f.read()

result = {
    'run_id': '${RUN_ID}',
    'dataset': '${DATASET}',
    'epochs': ${EPOCHS},
    'seed': ${SEED},
    'device': '${DEVICE}',
}

# 提取最终per-horizon结果
horizons = {}
for m in re.finditer(r'Evaluate best model.*horizon (\d+).*MAE: ([\d.]+).*RMSE: ([\d.]+).*MAPE: ([\d.]+)', content):
    h = int(m.group(1))
    horizons[f'h{h}'] = {
        'MAE': float(m.group(2)),
        'RMSE': float(m.group(3)),
        'MAPE': float(m.group(4))
    }
result['per_horizon'] = horizons

# 提取avg结果
avg_m = re.search(r'On average.*MAE: ([\d.]+).*RMSE: ([\d.]+).*MAPE: ([\d.]+)', content)
if avg_m:
    result['avg_MAE'] = float(avg_m.group(1))
    result['avg_RMSE'] = float(avg_m.group(2))
    result['avg_MAPE'] = float(avg_m.group(3))

# 提取Best Val
bv = re.search(r'Best Val\s*:\s*([\d.]+)', content)
if bv:
    result['best_val_mae'] = float(bv.group(1))

# 提取模型参数量
mp = re.search(r'Model params:\s*([\d,]+)', content)
if mp:
    result['model_params'] = int(mp.group(1).replace(',', ''))

# 提取最后一个DIAG快照
diag_lines = [l for l in content.split('\n') if '[DIAG' in l and 'adaptive_emb_gate' in l]
if diag_lines:
    gate_m = re.search(r'adaptive_emb_gate=([\d.]+)', diag_lines[-1])
    if gate_m:
        result['final_adaptive_gate'] = float(gate_m.group(1))

result_file = '${SAVE_DIR}/${RUN_ID}.json'
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\\nResults saved: {result_file}')
print(json.dumps(result, indent=2))
"

# ── 更新summary.json ──
python3 -c "
import json, os, glob

summary = {}
sf = 'experiments/results/summary.json'
if os.path.exists(sf):
    with open(sf) as f:
        summary = json.load(f)

# 读取本次结果
with open('experiments/results/${RUN_ID}.json') as f:
    this_run = json.load(f)

summary['${RUN_ID}'] = this_run

# 更新best
best_key = None
best_mae = 999
for k, v in summary.items():
    mae = v.get('avg_MAE')
    if mae is not None and mae < best_mae:
        best_mae = mae
        best_key = k
if best_key:
    summary['_best'] = best_key

with open(sf, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary updated. Best: {best_key} (MAE={best_mae})')
"

# ── 自动push (如果设置了GIT_TOKEN) ──
if [ -n "${GIT_TOKEN:-}" ]; then
    git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git"
    git add -A
    git commit -m "experiment: ${RUN_ID} — ${DATASET} seed=${SEED} epochs=${EPOCHS}" \
        --author="dylanyunlon <dogechat@163.com>" || true
    git push origin main
    echo "✓ Results pushed to git"
else
    echo "⚠ GIT_TOKEN not set — results saved locally only"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  DONE. Results: experiments/results/${RUN_ID}.json"
echo "  Log: ${LOG_FILE}"
echo "═══════════════════════════════════════════════"
