#!/usr/bin/env bash
# experiments/run_all.sh — 全流程: 基线 + Walpurgis 并行训练 + 评估 + push
# 在 ags1 上运行: nohup bash experiments/run_all.sh &
# GPU 分配: GPU0 (A6000) = D2STGNN baseline
#           GPU1 (A6000) = STAEFormer baseline
#           GPU2 (H100)  = Walpurgis (我们的模型, 需要最多显存)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
export GIT_TOKEN="${GIT_TOKEN:-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}"

# 配置 git 认证
git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" 2>/dev/null || true

echo "============================================"
echo " Walpurgis Full Experiment Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Server: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Phase 1: 环境准备 ──────────────────────────────────
echo ""
echo "[Phase 1] Environment setup..."
bash experiments/setup_env.sh

# ── Phase 2: 并行训练 ──────────────────────────────────
echo ""
echo "[Phase 2] Launching parallel training..."
PIDS=()

# GPU0: D2STGNN baseline
echo "  GPU0 (A6000): D2STGNN baseline..."
GPU=0 DATASET=METR-LA bash experiments/run_baselines.sh --model d2stgnn --gpu 0 \
    > experiments/results/baseline_d2stgnn.log 2>&1 &
PIDS+=($!)

# GPU2: Walpurgis (H100, 主实验)
echo "  GPU2 (H100):  Walpurgis..."
GPU=2 EPOCHS=100 TAG=walpurgis DATASET=METR-LA bash experiments/run_metrla.sh \
    --gpu 2 --epochs 100 --tag walpurgis --no-push \
    > experiments/results/walpurgis_metrla.log 2>&1 &
PIDS+=($!)

# 等待所有训练完成
echo ""
echo "Waiting for training jobs... PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "Job $pid finished with error"
done
echo "All training complete at $(date '+%H:%M:%S')"

# ── Phase 3: 汇总评估 ──────────────────────────────────
echo ""
echo "[Phase 3] Collecting results..."
python3 -c "
import json, glob, os

results = {}
for f in sorted(glob.glob('experiments/results/*/result.json')):
    with open(f) as fp:
        r = json.load(fp)
    name = r.get('run_id', os.path.basename(os.path.dirname(f)))
    results[name] = {
        'MAE': r.get('best_val_mae', 'N/A'),
        'RMSE': r.get('best_val_rmse', 'N/A'),
        'epoch': r.get('best_epoch', 'N/A'),
    }
    print(f'{name}: MAE={results[name][\"MAE\"]}')

# 写汇总
with open('experiments/results/summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSummary: experiments/results/summary.json')
"

# ── Phase 4: Git push ──────────────────────────────────
echo ""
echo "[Phase 4] Pushing results to git..."
cd "$REPO_DIR"
git add experiments/results/
git commit -m "experiment: METR-LA full run $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || echo "Push failed"

echo ""
echo "============================================"
echo " Pipeline complete!"
echo "============================================"
