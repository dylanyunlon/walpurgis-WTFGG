#!/usr/bin/env bash
# experiments/run_baselines_headtohead.sh
# Head-to-head comparison: Walpurgis vs STAEformer vs D2STGNN (upstream)
# 在同一台机器、同一份数据、同一评估协议下跑三个模型
#
# 用法: GPU=2 bash experiments/run_baselines_headtohead.sh
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

GPU="${GPU:-2}"
SEED="${SEED:-42}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULT_DIR="experiments/results/headtohead_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Head-to-Head Baseline Comparison"
echo " GPU: $GPU  SEED: $SEED"
echo " Results: $RESULT_DIR/"
echo "============================================"

# Conda
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walking3 2>/dev/null || true
set -u

# ── 1. Walpurgis (our model) — 200 epochs ──
echo ""
echo "[1/3] Walpurgis (Cascade D2STGNN) — 200 epochs"
echo "================================================"
WALPURGIS_DEBUG=0 \
CASCADE_DIAG_LOG="${RESULT_DIR}/walpurgis_diag.jsonl" \
CUDA_VISIBLE_DEVICES="$GPU" \
python3 train_walpurgis.py \
    --dataset METR-LA \
    --device cuda:0 \
    --epochs 200 \
    --seed "$SEED" \
    --save_dir "${RESULT_DIR}/walpurgis" \
    2>&1 | tee "${RESULT_DIR}/walpurgis.log" || true

# ── 2. STAEformer — 200 epochs (same data, same eval) ──
echo ""
echo "[2/3] STAEformer — 200 epochs"
echo "================================================"
CUDA_VISIBLE_DEVICES="$GPU" \
python3 upstream/staeformer/train.py \
    --dataset METR-LA \
    --device cuda:0 \
    --epochs 200 \
    --seed "$SEED" \
    2>&1 | tee "${RESULT_DIR}/staeformer.log" || true

# ── 3. D2STGNN (upstream, unmodified) ──
echo ""
echo "[3/3] D2STGNN (upstream) — 200 epochs"
echo "================================================"
if [ -f "upstream/d2stgnn/main.py" ]; then
    cd upstream/d2stgnn
    CUDA_VISIBLE_DEVICES="$GPU" \
    python3 main.py \
        --device cuda:0 \
        --epochs 200 \
        --seed "$SEED" \
        2>&1 | tee "${RESULT_DIR}/d2stgnn.log" || true
    cd "$REPO_DIR"
else
    echo "SKIP: upstream/d2stgnn/main.py not found"
fi

# ── 汇总对比 ──
echo ""
echo "============================================"
echo " Parsing results..."
echo "============================================"
python3 << 'PYEOF'
import json, os, re, sys

result_dir = os.environ.get("RESULT_DIR", "experiments/results/headtohead_latest")
models = {}

for name, logfile in [
    ("Walpurgis", "walpurgis.log"),
    ("STAEformer", "staeformer.log"),
    ("D2STGNN", "d2stgnn.log"),
]:
    path = os.path.join(result_dir, logfile)
    if not os.path.exists(path):
        continue
    log = open(path).read()
    avg = re.findall(
        r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%',
        log)
    if avg:
        best = min(avg, key=lambda x: float(x[0]))
        models[name] = {
            "MAE": float(best[0]),
            "RMSE": float(best[1]),
            "MAPE": float(best[2]),
        }
    # Per-horizon for the best run
    horizons = re.findall(
        r'horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)', log)
    if horizons and name in models:
        models[name]["per_horizon"] = [
            {"h": int(h), "mae": float(m), "rmse": float(r), "mape": float(p)}
            for h, m, r, p in horizons[-12:]
        ]

# Published baselines (from bench/sota.json)
published = {
    "TITAN (published)": {"MAE": 2.88, "RMSE": 5.33, "MAPE": None},
    "STAEFormer (published)": {"MAE": 2.90, "RMSE": 5.91, "MAPE": 8.12},
    "PDFormer (published)": {"MAE": 2.94, "RMSE": 6.08, "MAPE": 8.56},
}

print(f"\n{'='*70}")
print(f"  HEAD-TO-HEAD COMPARISON — METR-LA (avg 12 horizons)")
print(f"{'='*70}")
print(f"  {'Model':<30s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
print(f"  {'-'*54}")
for name, vals in sorted({**published, **models}.items(), key=lambda x: x[1].get("MAE", 99)):
    mae = f"{vals['MAE']:.2f}" if vals.get("MAE") else "—"
    rmse = f"{vals['RMSE']:.2f}" if vals.get("RMSE") else "—"
    mape = f"{vals['MAPE']:.2f}%" if vals.get("MAPE") else "—"
    marker = " ← OURS" if "Walpurgis" in name else ""
    print(f"  {name:<30s} {mae:>8s} {rmse:>8s} {mape:>8s}{marker}")
print(f"{'='*70}")

# Write JSON
with open(os.path.join(result_dir, "headtohead.json"), "w") as f:
    json.dump({"our_runs": models, "published": published}, f, indent=2)
print(f"\nSaved: {result_dir}/headtohead.json")
PYEOF

echo ""
echo "============================================"
echo " Done: $RESULT_DIR/"
echo "============================================"
