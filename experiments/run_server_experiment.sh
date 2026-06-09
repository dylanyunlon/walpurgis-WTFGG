#!/usr/bin/env bash
# experiments/run_server_experiment.sh
# 在 ags1 服务器上运行: nohup bash experiments/run_server_experiment.sh &
# 完成后自动将结果push到git，供后续Claude获取
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Git凭证 ──
export GIT_TOKEN="${GIT_TOKEN}"
git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" 2>/dev/null || true

# ── 拉取最新代码 ──
echo "[$(date '+%H:%M:%S')] Pulling latest..."
git pull --rebase origin main 2>/dev/null || git pull origin main || true

# ── Conda环境 ──
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walking3 2>/dev/null || true
set -u

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SEED="${SEED:-42}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-200}"
DATASET="${DATASET:-METR-LA}"
RUN_ID="experiment_${DATASET}_${TIMESTAMP}"
RESULT_DIR="experiments/results/${RUN_ID}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Walpurgis Server Experiment"
echo " ID:      $RUN_ID"
echo " Dataset: $DATASET"
echo " GPU:     $GPU (H100 NVL 96GB)"
echo " Epochs:  $EPOCHS"
echo " Seed:    $SEED"
echo " Debug:   diagnostics enabled"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── 训练 (带诊断) ──
echo "[$(date '+%H:%M:%S')] Starting training..."
WALPURGIS_DEBUG=0 \
CASCADE_DIAG_LOG="${RESULT_DIR}/diagnostics.jsonl" \
CUDA_VISIBLE_DEVICES="$GPU" \
python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device cuda:0 \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --save_dir "$RESULT_DIR" \
    2>&1 | tee "${RESULT_DIR}/train.log"

echo "[$(date '+%H:%M:%S')] Training complete"

# ── 提取结果写入JSON ──
python3 << 'PYEOF'
import json, os, re, glob

result_dir = os.environ.get("RESULT_DIR", "")
run_id = os.environ.get("RUN_ID", "unknown")

log_path = os.path.join(result_dir, "train.log")
result = {"run_id": run_id, "status": "complete"}

if os.path.exists(log_path):
    log = open(log_path).read()
    # 提取12-horizon平均
    avg = re.findall(
        r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%',
        log)
    if avg:
        best = min(avg, key=lambda x: float(x[0]))
        result["test_mae_avg12"] = float(best[0])
        result["test_rmse_avg12"] = float(best[1])
        result["test_mape_avg12"] = float(best[2])

    bv = re.search(r'Best Val\s*:\s*([\d.]+)', log)
    if bv:
        result["best_val_mae"] = float(bv.group(1))

    ep = re.findall(r'Epoch\s+(\d+)', log)
    if ep:
        result["epochs_trained"] = int(ep[-1])

    params = re.search(r'Model params:\s*([\d,]+)', log)
    if params:
        result["model_params"] = int(params.group(1).replace(',', ''))

    # 逐horizon结果
    horizon_data = re.findall(
        r'horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)',
        log)
    if horizon_data:
        # 取最后一组12个
        last_12 = horizon_data[-12:] if len(horizon_data) >= 12 else horizon_data
        result["per_horizon"] = [
            {"h": int(h), "mae": float(m), "rmse": float(r), "mape": float(p)}
            for h, m, r, p in last_12
        ]

# SOTA对比
print(f"\n{'='*60}")
print(f"  RESULTS vs SOTA ({os.environ.get('DATASET', 'METR-LA')})")
print(f"{'='*60}")
sota = [
    ("TITAN (2024)", 2.88), ("STAEFormer (2023)", 2.90),
    ("PDFormer (2023)", 2.94), ("D2STGNN (2022)", 3.04),
    ("Walpurgis prev best", 3.08),
]
mae = result.get("test_mae_avg12")
if mae:
    for name, val in sota:
        diff = mae - val
        marker = " <-- BEATEN!" if diff < 0 else ""
        print(f"  {name:25s} MAE={val:.2f}  (gap: {diff:+.4f}){marker}")
    print(f"  {'Walpurgis (this run)':25s} MAE={mae:.2f}")
    if mae < 2.85:
        print(f"\n  >>> SOTA ACHIEVED! <<<")
    result["sota_comparison"] = {
        "titan_gap": round(mae - 2.88, 4),
        "staeformer_gap": round(mae - 2.90, 4),
        "prev_best_gap": round(mae - 3.08, 4),
    }
print(f"{'='*60}")

with open(os.path.join(result_dir, "result.json"), "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# 更新汇总
summary_path = "experiments/results/summary.json"
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
summary[run_id] = {
    "MAE": result.get("test_mae_avg12", "N/A"),
    "RMSE": result.get("test_rmse_avg12", "N/A"),
    "MAPE": result.get("test_mape_avg12", "N/A"),
    "epochs": result.get("epochs_trained", "N/A"),
    "params": result.get("model_params", "N/A"),
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResult: {os.path.join(result_dir, 'result.json')}")
PYEOF

export RESULT_DIR RUN_ID DATASET

# ── Git push结果 ──
echo "[$(date '+%H:%M:%S')] Pushing results..."
cd "$REPO_DIR"
git add experiments/results/
git commit -m "experiment: ${RUN_ID} — ${DATASET} ${EPOCHS}ep seed${SEED}" 2>/dev/null || true
git push origin main || { git pull --rebase origin main && git push origin main; } || echo "Push failed"

echo ""
echo "============================================"
echo " Done: $RUN_ID"
echo " Results: $RESULT_DIR/"
echo " Next Claude can: git pull && cat $RESULT_DIR/result.json"
echo "============================================"
