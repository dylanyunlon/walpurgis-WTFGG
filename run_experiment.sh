#!/usr/bin/env bash
# run_experiment.sh — 在GPU服务器上运行完整实验, 自动评估+push结果
# 使用: ./run_experiment.sh <variant> <dataset> <device> [epochs]
# 例如: ./run_experiment.sh cathexis METR-LA cuda:0 80
set -euo pipefail

VARIANT="${1:?Usage: $0 <variant> <dataset> <device> [epochs]}"
DATASET="${2:?Usage: $0 <variant> <dataset> <device> [epochs]}"
DEVICE="${3:-cuda:0}"
EPOCHS="${4:-80}"
SEEDS="${5:-42,123,456}"  # 3 seeds for D10 stability metric

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

TOKEN="${GITHUB_TOKEN}"
REMOTE_URL="https://${TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git"

echo "========================================================"
echo " Walpurgis Experiment Pipeline"
echo " Variant: ${VARIANT} | Dataset: ${DATASET}"
echo " Device: ${DEVICE} | Epochs: ${EPOCHS} | Seeds: ${SEEDS}"
echo " $(date)"
echo "========================================================"

# Activate conda if available
conda activate walpurgis 2>/dev/null || true

# Pull latest
git pull origin main 2>/dev/null || true

RESULTS_DIR="output/experiment_results"
mkdir -p "$RESULTS_DIR"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
ALL_MAES=()

for SEED in "${SEED_ARRAY[@]}"; do
    echo ""
    echo "--- Training seed=${SEED} ---"
    
    # Set random seed
    export PYTHONHASHSEED=$SEED
    
    # Run training
    python3 "train_${VARIANT}.py" \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        2>&1 | tee "${RESULTS_DIR}/${VARIANT}_${DATASET}_seed${SEED}.log"
    
    # Evaluate and save results
    python3 << PYEOF
import torch, json, sys, os, time, numpy as np
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'src'))

# Load best model and run evaluation
model_path = "output/D2STGNN_${DATASET}.pt"
if not os.path.exists(model_path):
    model_path = "output/D2STGNN_CATHEXIS_${DATASET}.pt"
    
results = {
    "variant": "${VARIANT}",
    "dataset": "${DATASET}",
    "seed": ${SEED},
    "epochs": ${EPOCHS},
    "device": "${DEVICE}",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Parse the last test output from log
log_path = "${RESULTS_DIR}/${VARIANT}_${DATASET}_seed${SEED}.log"
with open(log_path) as f:
    lines = f.readlines()

horizons = []
for line in lines:
    if line.startswith("Horizon "):
        parts = line.strip().split(",")
        h = int(parts[0].split()[1].rstrip(","))
        mae = float(parts[1].split(":")[1])
        rmse = float(parts[2].split(":")[1])
        mape = float(parts[3].split(":")[1])
        horizons.append({"horizon": h, "MAE": mae, "RMSE": rmse, "MAPE": mape})
    if "Avg MAE" in line:
        parts = line.strip().split("|")
        results["avg_MAE"] = float(parts[0].split(":")[1])
        results["avg_RMSE"] = float(parts[1].split(":")[1])
        results["avg_MAPE"] = float(parts[2].split(":")[1].replace("%",""))

if horizons:
    results["per_horizon"] = horizons
    results["MAE_15min"] = horizons[2]["MAE"] if len(horizons) > 2 else None
    results["MAE_30min"] = horizons[5]["MAE"] if len(horizons) > 5 else None
    results["MAE_60min"] = horizons[11]["MAE"] if len(horizons) > 11 else None

out_path = "${RESULTS_DIR}/${VARIANT}_${DATASET}_seed${SEED}_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results saved: {out_path}")
print(f"  Avg MAE: {results.get('avg_MAE', 'N/A')}")
PYEOF
done

# Aggregate multi-seed results
python3 << PYEOF
import json, glob, numpy as np

pattern = "${RESULTS_DIR}/${VARIANT}_${DATASET}_seed*_results.json"
files = sorted(glob.glob(pattern))
if not files:
    print("No result files found")
    exit(0)

maes = []
all_results = []
for f in files:
    r = json.load(open(f))
    all_results.append(r)
    if r.get("avg_MAE"): maes.append(r["avg_MAE"])

summary = {
    "variant": "${VARIANT}",
    "dataset": "${DATASET}",
    "num_seeds": len(files),
    "seeds": [r.get("seed") for r in all_results],
    "avg_MAE_mean": float(np.mean(maes)) if maes else None,
    "avg_MAE_std": float(np.std(maes)) if len(maes)>1 else None,
    "individual_results": all_results,
}

# Best seed results
if maes:
    best_idx = np.argmin(maes)
    summary["best_seed"] = all_results[best_idx]

out_path = "${RESULTS_DIR}/${VARIANT}_${DATASET}_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Summary: {out_path}")
print(f"  Seeds: {len(files)}, Mean MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}" if maes else "No MAE data")
PYEOF

# Push results to git
echo ""
echo "--- Pushing results to git ---"
git add -A "$RESULTS_DIR/" "output/*.pt" "log/"
git commit -m "result(${VARIANT}): ${DATASET} experiment — seeds ${SEEDS}" 2>/dev/null || echo "Nothing to commit"
git remote set-url origin "$REMOTE_URL" 2>/dev/null || true
git push origin main 2>/dev/null || echo "Push failed (will retry)"

echo ""
echo "========================================================"
echo " Experiment complete!"
echo " Results in: ${RESULTS_DIR}/"
echo "========================================================"
