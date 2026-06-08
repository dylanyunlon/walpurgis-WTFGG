#!/bin/bash
# gpu_experiment.sh — 在GPU服务器上运行实验并自动push结果到git
# 供子Claude在GPU服务器上执行
# Usage: bash gpu_experiment.sh <variant> <dataset> [device] [epochs]
# Example: bash gpu_experiment.sh reverie METR-LA cuda:0 80
set -e

VARIANT=${1:?Usage: bash gpu_experiment.sh <variant> <dataset> [device] [epochs]}
DATASET=${2:?Usage: bash gpu_experiment.sh <variant> <dataset> [device] [epochs]}
DEVICE=${3:-cuda:0}
EPOCHS=${4:-80}
SEEDS=${5:-"42 123 456"}

REPO_URL="https://${GH_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git"
WORK_DIR="/tmp/walpurgis_exp_${VARIANT}_${DATASET}"

echo "============================================"
echo "  GPU Experiment Runner"
echo "  Variant : walpurgis_${VARIANT}"
echo "  Dataset : ${DATASET}"
echo "  Device  : ${DEVICE}"
echo "  Epochs  : ${EPOCHS}"
echo "  Seeds   : ${SEEDS}"
echo "============================================"

# 1. Clone/pull latest
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 2. Setup conda env (reuse if exists)
if command -v conda &>/dev/null; then
    if conda env list | grep -q "walpurgis"; then
        echo "[GPU] Using existing conda env: walpurgis"
        eval "$(conda shell.bash hook)"
        conda activate walpurgis
    else
        echo "[GPU] Creating conda env: walpurgis"
        conda create -n walpurgis python=3.10 -y
        eval "$(conda shell.bash hook)"
        conda activate walpurgis
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install numpy scipy pyyaml scikit-learn
    fi
else
    echo "[GPU] No conda found, using system Python"
fi

# 3. Check dataset exists
if [ ! -d "datasets/${DATASET}" ]; then
    echo "[GPU] ERROR: datasets/${DATASET} not found!"
    echo "[GPU] For METR-LA/PEMS-BAY, download from upstream/d2stgnn/datasets/"
    echo "[GPU] Run: python upstream/d2stgnn/datasets/raw_data/${DATASET}/generate_training_data.py"
    exit 1
fi

# 4. Run experiments with multiple seeds
RESULT_DIR="output/results_${VARIANT}_${DATASET}"
mkdir -p "$RESULT_DIR"

for SEED in $SEEDS; do
    echo ""
    echo "[GPU] === Seed ${SEED} ==="
    SEED_DIR="${RESULT_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    python train_${VARIANT}.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        2>&1 | tee "${SEED_DIR}/train.log"

    # Copy model checkpoint
    cp output/D2STGNN_*_${DATASET}*.pt "${SEED_DIR}/" 2>/dev/null || true

    echo "[GPU] Seed ${SEED} complete"
done

# 5. Parse results and create summary
python3 -c "
import os, json, re, numpy as np

result_dir = '${RESULT_DIR}'
seeds = '${SEEDS}'.split()
all_mae, all_rmse, all_mape = [], [], []

for seed in seeds:
    log_path = os.path.join(result_dir, f'seed_{seed}', 'train.log')
    if not os.path.exists(log_path):
        continue
    with open(log_path) as f:
        lines = f.readlines()
    # Find last 'Avg MAE' line
    for line in reversed(lines):
        m = re.search(r'Avg MAE:\s*([\d.]+).*Avg RMSE:\s*([\d.]+).*Avg MAPE:\s*([\d.]+)', line)
        if m:
            all_mae.append(float(m.group(1)))
            all_rmse.append(float(m.group(2)))
            all_mape.append(float(m.group(3)))
            break

summary = {
    'variant': 'walpurgis_${VARIANT}',
    'dataset': '${DATASET}',
    'device': '${DEVICE}',
    'epochs': ${EPOCHS},
    'seeds': seeds,
    'results': {
        'MAE':  {'mean': np.mean(all_mae), 'std': np.std(all_mae), 'all': all_mae},
        'RMSE': {'mean': np.mean(all_rmse), 'std': np.std(all_rmse), 'all': all_rmse},
        'MAPE': {'mean': np.mean(all_mape), 'std': np.std(all_mape), 'all': all_mape},
    }
}
with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f\"\\n{'='*50}\")
print(f\"  RESULTS: walpurgis_${VARIANT} on ${DATASET}\")
print(f\"  MAE:  {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}\")
print(f\"  RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}\")
print(f\"  MAPE: {np.mean(all_mape):.4f} ± {np.std(all_mape):.4f}\")
print(f\"{'='*50}\")
"

# 6. Auto-push results to git
git add "${RESULT_DIR}/" output/*.json log/ 2>/dev/null || true
git commit -m "exp: walpurgis_${VARIANT} on ${DATASET} — $(date +%Y%m%d_%H%M)" || true
git push origin main

echo ""
echo "[GPU] Results pushed to git. Sub-claudes can pull to read."
echo "[GPU] Done."
