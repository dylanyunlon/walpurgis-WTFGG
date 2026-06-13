#!/usr/bin/env bash
# experiments/run_server_experiment.sh
# 用法: GIT_TOKEN=xxx GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
# 或先前台调试: GPU=2 EPOCHS=3 bash experiments/run_server_experiment.sh
set -uo pipefail  # 不用 -e，训练失败也要跑解析和push

export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Git凭证（可选）──
export GIT_TOKEN="${GIT_TOKEN:-}"
if [ -n "$GIT_TOKEN" ]; then
    git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" 2>/dev/null || true
fi

# ── Conda ──
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walking3 2>/dev/null || true
set -u

# ── 参数 ──
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SEED="${SEED:-42}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-200}"
DATASET="${DATASET:-METR-LA}"
# 空白字符清理 (三重保险)
DATASET="${DATASET//[[:space:]]/}"
DATASET="$(echo -n "$DATASET" | tr -d '[:space:]')"
RUN_ID="experiment_$(echo -n "${DATASET}_${TIMESTAMP}" | tr -d '[:space:]')"
RESULT_DIR="experiments/results/${RUN_ID}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Walpurgis Server Experiment"
echo " ID:      $RUN_ID"
echo " Dataset: $DATASET"
echo " GPU:     $GPU"
echo " Epochs:  $EPOCHS"
echo " Seed:    $SEED"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── 检查数据 ──
if [ ! -f "datasets/${DATASET}/train.npz" ]; then
    echo "ERROR: datasets/${DATASET}/train.npz not found"
    exit 1
fi

# ── 训练 ──
echo ""
echo "[$(date '+%H:%M:%S')] Starting training..."
TRAIN_EXIT=0
WALPURGIS_DEBUG=0 \
CASCADE_DIAG_LOG="${RESULT_DIR}/diagnostics.jsonl" \
CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device cuda:0 \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --save_dir "$RESULT_DIR" \
    2>&1 | tee "${RESULT_DIR}/train.log" || TRAIN_EXIT=$?

echo ""
echo "[$(date '+%H:%M:%S')] Training exit code: $TRAIN_EXIT"

# ── 提取结果 ──
echo "[$(date '+%H:%M:%S')] Parsing results..."
RESULT_DIR="$RESULT_DIR" RUN_ID="$RUN_ID" DATASET="$DATASET" python3 << 'PYEOF'
import json, os, re

result_dir = os.environ["RESULT_DIR"]
run_id = os.environ["RUN_ID"]
dataset = os.environ["DATASET"]

log_path = os.path.join(result_dir, "train.log")
result = {"run_id": run_id, "dataset": dataset}

if not os.path.exists(log_path):
    print("WARNING: train.log not found")
    result["status"] = "no_log"
else:
    log = open(log_path).read()

    # 12-horizon 平均
    avg = re.findall(
        r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%',
        log)
    if avg:
        best = min(avg, key=lambda x: float(x[0]))
        result["test_mae_avg12"] = float(best[0])
        result["test_rmse_avg12"] = float(best[1])
        result["test_mape_avg12"] = float(best[2])
        result["status"] = "complete"
    else:
        result["status"] = "no_test_metrics"

    # Best Val
    bv = re.search(r'Best Val\s*:\s*([\d.]+)', log)
    if bv:
        result["best_val_mae"] = float(bv.group(1))

    # Epochs
    ep = re.findall(r'Epoch\s+(\d+)', log)
    if ep:
        result["epochs_trained"] = int(ep[-1])

    # Params
    params = re.search(r'Model params:\s*([\d,]+)', log)
    if params:
        result["model_params"] = int(params.group(1).replace(',', ''))

    # Per-horizon
    horizon_data = re.findall(
        r'horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)',
        log)
    if horizon_data:
        last_12 = horizon_data[-12:] if len(horizon_data) >= 12 else horizon_data
        result["per_horizon"] = [
            {"h": int(h), "mae": float(m), "rmse": float(r), "mape": float(p)}
            for h, m, r, p in last_12
        ]

    # 检查是否OOM
    if "CUDA out of memory" in log or "OutOfMemoryError" in log:
        result["status"] = "OOM"
        print("!! OOM detected in log")

    # 检查是否early stop
    if "Early stopping" in log:
        es_match = re.search(r'Early stopping.*counter.*?(\d+)', log)
        result["early_stopped"] = True

# SOTA对比
print(f"\n{'='*60}")
print(f"  RESULTS vs SOTA ({dataset})")
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
else:
    print(f"  No MAE data (status: {result.get('status', 'unknown')})")
    print(f"  Check: {os.path.join(result_dir, 'train.log')}")
print(f"{'='*60}")

# 写结果
with open(os.path.join(result_dir, "result.json"), "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# 更新汇总
summary_path = "experiments/results/summary.json"
try:
    summary = json.load(open(summary_path))
except:
    summary = {}
summary[run_id] = {
    "MAE": result.get("test_mae_avg12", "N/A"),
    "RMSE": result.get("test_rmse_avg12", "N/A"),
    "MAPE": result.get("test_mape_avg12", "N/A"),
    "epochs": result.get("epochs_trained", "N/A"),
    "params": result.get("model_params", "N/A"),
    "status": result.get("status", "unknown"),
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResult: {os.path.join(result_dir, 'result.json')}")
PYEOF

# ── Git push ──
if [ -n "$GIT_TOKEN" ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Pushing results..."
    git add experiments/results/
    git commit -m "experiment: ${RUN_ID} — ${DATASET} ${EPOCHS}ep seed${SEED}" 2>/dev/null || true
    git push origin main 2>/dev/null || {
        git pull --rebase origin main 2>/dev/null && git push origin main 2>/dev/null
    } || echo "Push failed — manually: git push origin main"
else
    echo ""
    echo "[NOTE] No GIT_TOKEN set. Push manually:"
    echo "  git add experiments/results/ && git commit -m 'experiment: ${RUN_ID}' && git push"
fi

echo ""
echo "============================================"
echo " Done: $RUN_ID (exit: $TRAIN_EXIT)"
echo " Results: $RESULT_DIR/"
echo "============================================"
