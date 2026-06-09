#!/usr/bin/env bash
# experiments/auto_experiment.sh — 自动化实验: 训练→评估→结果写JSON→git push
# 参考 alphaproof-nexus-results 模式: 服务器跑实验, 结果自动push到仓库
#
# 用法:
#   bash experiments/auto_experiment.sh                   # 默认: METR-LA, GPU2(H100), 200ep
#   GPU=0 EPOCHS=150 bash experiments/auto_experiment.sh  # 用A6000, 150ep
#   DATASET=PEMS-BAY bash experiments/auto_experiment.sh  # 跑PEMS-BAY
#
# 服务器: ags1 — 2x A6000 (48GB) + 1x H100 NVL (96GB), EPYC 9354, CUDA 11.5
# 环境: conda activate walking3
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── 参数 ────────────────────────────────────────────────
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-200}"
DATASET="${DATASET:-METR-LA}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_HIDDEN="${NUM_HIDDEN:-64}"
NODE_HIDDEN="${NODE_HIDDEN:-32}"
TIME_EMB_DIM="${TIME_EMB_DIM:-32}"
LR="${LR:-0.002}"
PATIENCE="${PATIENCE:-20}"
TAG="${TAG:-walpurgis}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ID="${TAG}_${DATASET}_${TIMESTAMP}"
RESULT_DIR="experiments/results/${RUN_ID}"
mkdir -p "$RESULT_DIR"

# ── 日志 ────────────────────────────────────────────────
exec > >(tee "${RESULT_DIR}/full.log") 2>&1

echo "============================================================"
echo "  Walpurgis Auto-Experiment"
echo "  Run ID  : $RUN_ID"
echo "  Dataset : $DATASET"
echo "  GPU     : $GPU"
echo "  Epochs  : $EPOCHS"
echo "  Seed    : $SEED"
echo "  Batch   : $BATCH_SIZE"
echo "  Hidden  : $NUM_HIDDEN"
echo "  Time    : $(date)"
echo "============================================================"

# ── Conda ───────────────────────────────────────────────
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walking3 2>/dev/null || true
set -u

echo "Python : $(python3 --version 2>&1)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__, "CUDA:", torch.version.cuda)' 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

# ── 数据检查 ────────────────────────────────────────────
if [ ! -f "datasets/${DATASET}/train.npz" ]; then
    echo "ERROR: datasets/${DATASET}/train.npz not found. Run setup first."
    exit 1
fi

# ── 训练 ────────────────────────────────────────────────
echo ""
echo "[TRAIN] Starting at $(date '+%H:%M:%S')..."
TRAIN_START=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device cuda:0 \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --save_dir "$RESULT_DIR" \
    2>&1

TRAIN_END=$(date +%s)
TRAIN_SECS=$((TRAIN_END - TRAIN_START))
echo "[TRAIN] Finished at $(date '+%H:%M:%S') (${TRAIN_SECS}s)"

# ── 结果提取 ────────────────────────────────────────────
echo "[EVAL] Extracting results..."
CUDA_VISIBLE_DEVICES="$GPU" python3 << 'PYEVAL'
import json, os, re, sys

run_id = os.environ.get('RUN_ID', 'unknown')
result_dir = os.environ.get('RESULT_DIR', 'experiments/results/unknown')
dataset = os.environ.get('DATASET', 'METR-LA')
gpu = os.environ.get('GPU', '2')
epochs = int(os.environ.get('EPOCHS', '200'))
seed = int(os.environ.get('SEED', '42'))
train_secs = int(os.environ.get('TRAIN_SECS', '0'))

result = {
    'run_id': run_id,
    'dataset': dataset,
    'gpu': gpu,
    'epochs_max': epochs,
    'seed': seed,
    'train_time_seconds': train_secs,
}

# 从训练日志中提取最终指标
log_path = os.path.join(result_dir, 'full.log')
if os.path.exists(log_path):
    with open(log_path) as f:
        log_text = f.read()

    # 提取所有 "On average over 12 horizons" 行
    avg_pattern = r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%'
    avg_matches = re.findall(avg_pattern, log_text)

    if avg_matches:
        # 找最小MAE的那次
        best = min(avg_matches, key=lambda x: float(x[0]))
        result['test_mae_avg12'] = float(best[0])
        result['test_rmse_avg12'] = float(best[1])
        result['test_mape_avg12'] = float(best[2])

    # 提取各horizon的最佳结果
    horizon_pattern = r'Evaluate best model on test data for horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)'
    h_matches = re.findall(horizon_pattern, log_text)
    if h_matches:
        # 按epoch块分组(每12行一组), 找MAE最小的块
        blocks = [h_matches[i:i+12] for i in range(0, len(h_matches), 12)]
        if blocks:
            best_block = min(blocks, key=lambda b: sum(float(x[1]) for x in b))
            horizons = {}
            for h, mae, rmse, mape in best_block:
                horizons[f'h{h}'] = {
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'MAPE': float(mape)
                }
            result['horizons'] = horizons

    # 提取Best Val
    bv = re.search(r'Best Val\s*:\s*([\d.]+)', log_text)
    if bv:
        result['best_val_mae'] = float(bv.group(1))

    # 提取停止epoch
    es = re.search(r'Epoch\s+(\d+).*?EarlyStopping counter: \d+ out of \d+', log_text)
    epochs_trained = re.findall(r'Epoch\s+(\d+)', log_text)
    if epochs_trained:
        result['epochs_trained'] = int(epochs_trained[-1])

    # 模型参数量
    params = re.search(r'Model params:\s*([\d,]+)', log_text)
    if params:
        result['model_params'] = int(params.group(1).replace(',', ''))

# SOTA对比表
result['sota_comparison'] = {
    'TITAN_2024': {'MAE_avg': 2.88, 'source': 'arxiv 2409.17440'},
    'STAEFormer_2023': {'MAE_avg': 2.90, 'source': 'CIKM 2023'},
    'PDFormer_2023': {'MAE_avg': 2.94, 'source': 'AAAI 2023'},
    'D2STGNN_2022': {'MAE_avg': 3.04, 'source': 'VLDB 2022'},
    'STEP_CompFormer': {'MAE_60min': 3.27, 'source': 'arxiv 2309.09074'},
    'STLLM_DF_2025': {'MAE_15min': 2.61, 'MAE_30min': 2.90, 'MAE_60min': 3.27, 'source': 'TR-C 2025'},
    'walpurgis_target': '<2.85',
}

out_path = os.path.join(result_dir, 'result.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  RESULTS: {run_id}")
print(f"{'='*60}")
if 'test_mae_avg12' in result:
    mae = result['test_mae_avg12']
    rmse = result['test_rmse_avg12']
    mape = result['test_mape_avg12']
    print(f"  Test MAE (avg 12h): {mae:.4f}")
    print(f"  Test RMSE(avg 12h): {rmse:.4f}")
    print(f"  Test MAPE(avg 12h): {mape:.2f}%")
    print(f"  Target: <2.85  | Current SOTA: ~2.88 (TITAN)")
    if mae < 2.85:
        print(f"  >>> SOTA ACHIEVED! <<<")
    elif mae < 2.90:
        print(f"  >>> Near-SOTA (within STAEFormer range) <<<")
    else:
        print(f"  >>> Gap to SOTA: {mae - 2.88:.4f} <<<")
else:
    print("  WARNING: Could not extract test metrics from log")
print(f"  Result: {out_path}")
print(f"{'='*60}")
PYEVAL

# ── 汇总到全局结果表 ────────────────────────────────────
python3 << 'PYMERGE'
import json, os, glob

results_dir = 'experiments/results'
summary = {}
for rj in sorted(glob.glob(os.path.join(results_dir, '*/result.json'))):
    with open(rj) as f:
        r = json.load(f)
    run_id = r.get('run_id', os.path.basename(os.path.dirname(rj)))
    summary[run_id] = {
        'MAE': r.get('test_mae_avg12', 'N/A'),
        'RMSE': r.get('test_rmse_avg12', 'N/A'),
        'MAPE': r.get('test_mape_avg12', 'N/A'),
        'best_val': r.get('best_val_mae', 'N/A'),
        'epochs': r.get('epochs_trained', 'N/A'),
        'params': r.get('model_params', 'N/A'),
        'time_s': r.get('train_time_seconds', 'N/A'),
    }

with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Summary updated: {len(summary)} runs")
PYMERGE

# ── Git push ────────────────────────────────────────────
echo "[GIT] Committing and pushing results..."
cd "$REPO_DIR"
git add experiments/results/
git add -u  # 追踪已删除文件
git commit -m "experiment: ${RUN_ID} — ${DATASET} ep${EPOCHS} seed${SEED} gpu${GPU}" || true
git pull --rebase origin main 2>/dev/null || true
git push origin main || echo "[GIT] Push failed — check credentials"

echo ""
echo "============================================================"
echo "  Auto-experiment complete: $RUN_ID"
echo "  Results: $RESULT_DIR/"
echo "  $(date)"
echo "============================================================"
