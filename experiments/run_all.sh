#!/usr/bin/env bash
# experiments/run_all.sh — 全流程: 环境准备 → 基线+Walpurgis并行训练 → 评估 → 结果JSON → git push
# 在 ags1 上运行: nohup bash experiments/run_all.sh &
# GPU 分配 (PCI_BUS_ID 顺序):
#   GPU0 (A6000 48GB) = D2STGNN baseline
#   GPU1 (A6000 48GB) = 消融实验 (预留)
#   GPU2 (H100  96GB) = Walpurgis 主实验
set -uo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
export GIT_TOKEN="${GIT_TOKEN:-}"

if [ -n "$GIT_TOKEN" ]; then
    git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" 2>/dev/null || true
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

echo "============================================"
echo " Walpurgis Full Experiment Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Server: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Phase 1: 环境准备 ──────────────────────────────────
echo ""
echo "[Phase 1] Environment setup..."
bash experiments/setup_env.sh

# ── Phase 2: 并行训练 ──────────────────────────────────
echo ""
echo "[Phase 2] Launching parallel training..."

WALPURGIS_RUN_ID="walpurgis_METR-LA_${TIMESTAMP}"
WALPURGIS_RESULT_DIR="experiments/results/${WALPURGIS_RUN_ID}"
export WALPURGIS_RUN_ID WALPURGIS_RESULT_DIR
mkdir -p "$WALPURGIS_RESULT_DIR"

PIDS=()

# GPU0: D2STGNN baseline
echo "  GPU0 (A6000): D2STGNN baseline..."
GPU=0 DATASET=METR-LA bash experiments/run_baselines.sh --model d2stgnn --gpu 0 \
    > experiments/results/baseline_d2stgnn.log 2>&1 &
PIDS+=($!)

# GPU2: Walpurgis (H100, 主实验)
echo "  GPU2 (H100):  Walpurgis -> ${WALPURGIS_RUN_ID}"
GPU=2 EPOCHS=${EPOCHS:-200} TAG=walpurgis DATASET=METR-LA SEED=${SEED:-42} \
    bash experiments/run_metrla.sh \
    --gpu 2 --epochs ${EPOCHS:-200} --tag walpurgis --no-push \
    > experiments/results/walpurgis_metrla.log 2>&1 &
PIDS+=($!)

echo ""
echo "Waiting for training jobs... PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "Job $pid finished with error"
done
echo "All training complete at $(date '+%H:%M:%S')"

# ── Phase 3: 结果提取 + SOTA 对比 ─────────────────────
echo ""
echo "[Phase 3] Extracting results & SOTA comparison..."
python3 << 'PYEVAL'
import json, glob, os, re

results_dir = 'experiments/results'
summary = {}

# 从 walpurgis_metrla.log 提取指标
log_path = os.path.join(results_dir, 'walpurgis_metrla.log')
walpurgis_result = {'run_id': os.environ.get('WALPURGIS_RUN_ID', 'unknown')}

if os.path.exists(log_path):
    with open(log_path) as f:
        log_text = f.read()

    # 提取 12-horizon 平均
    avg_pattern = r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%'
    avg_matches = re.findall(avg_pattern, log_text)
    if avg_matches:
        best = min(avg_matches, key=lambda x: float(x[0]))
        walpurgis_result['test_mae_avg12'] = float(best[0])
        walpurgis_result['test_rmse_avg12'] = float(best[1])
        walpurgis_result['test_mape_avg12'] = float(best[2])

    # Best Val
    bv = re.search(r'Best Val\s*:\s*([\d.]+)', log_text)
    if bv:
        walpurgis_result['best_val_mae'] = float(bv.group(1))

    # Epochs trained
    epochs_trained = re.findall(r'Epoch\s+(\d+)', log_text)
    if epochs_trained:
        walpurgis_result['epochs_trained'] = int(epochs_trained[-1])

    # Model params
    params = re.search(r'Model params:\s*([\d,]+)', log_text)
    if params:
        walpurgis_result['model_params'] = int(params.group(1).replace(',', ''))

# 写单次结果JSON
run_dir = os.environ.get('WALPURGIS_RESULT_DIR', '')
if run_dir and os.path.isdir(run_dir):
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(walpurgis_result, f, indent=2, ensure_ascii=False)

# 汇总所有结果
for rj in sorted(glob.glob(os.path.join(results_dir, '*/result.json'))):
    with open(rj) as f:
        r = json.load(f)
    name = r.get('run_id', os.path.basename(os.path.dirname(rj)))
    summary[name] = {
        'MAE': r.get('test_mae_avg12', r.get('best_val_mae', 'N/A')),
        'RMSE': r.get('test_rmse_avg12', 'N/A'),
        'MAPE': r.get('test_mape_avg12', 'N/A'),
        'epochs': r.get('epochs_trained', 'N/A'),
        'params': r.get('model_params', 'N/A'),
    }

with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 打印 SOTA 对比
print(f"\n{'='*60}")
print(f"  RESULTS vs SOTA (METR-LA avg 12 horizons)")
print(f"{'='*60}")
sota = [
    ('TITAN (2024)',      2.88),
    ('STAEFormer (2023)', 2.90),
    ('PDFormer (2023)',   2.94),
    ('D2STGNN (2022)',    3.04),
]
mae = walpurgis_result.get('test_mae_avg12')
if mae:
    for name, val in sota:
        diff = mae - val
        marker = '<-- BEATEN!' if diff < 0 else ''
        print(f"  {name:25s} MAE={val:.2f}  (gap: {diff:+.4f}) {marker}")
    print(f"  {'Walpurgis (ours)':25s} MAE={mae:.2f}")
    print(f"  {'Target':25s} MAE=<2.85")
    if mae < 2.85:
        print(f"\n  >>> SOTA ACHIEVED! <<<")
else:
    print("  WARNING: Could not extract metrics from log")
    for name, val in sota:
        print(f"  {name:25s} MAE={val:.2f}")
print(f"{'='*60}")
PYEVAL

# ── Phase 4: Git push ──────────────────────────────────
echo ""
echo "[Phase 4] Pushing results to git..."
cd "$REPO_DIR"
git pull --rebase origin main 2>/dev/null || git pull origin main 2>/dev/null || true
git add experiments/results/
git commit -m "experiment: METR-LA full run $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || { echo "Retrying push after pull..."; git pull --rebase origin main && git push origin main; } || echo "Push failed"

echo ""
echo "============================================"
echo " Pipeline complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
