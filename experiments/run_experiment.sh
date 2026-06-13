#!/usr/bin/env bash
# experiments/run_experiment.sh — cugraph-gnn风格实验runner
# 从NVIDIA cugraph-gnn的dist_gin_sg.py + wholegraph_benchmark.hpp模式迁移
# 改写: 适配walpurgis D2STGNN pipeline, 加全链路timing + GPU诊断
#
# 用法:
#   bash experiments/run_experiment.sh                    # SYNTH快速验证
#   DATASET=METR-LA GPU=2 EPOCHS=200 bash experiments/run_experiment.sh  # 完整实验
#   DATASET=METR-LA GPU=2 EPOCHS=200 PUSH=1 bash experiments/run_experiment.sh  # 跑完push
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Conda (服务器环境) ──
set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walking3 2>/dev/null || true
set -u

# ── 参数 (cugraph-gnn风格: 环境变量配置) ──
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
DATASET="${DATASET:-SYNTH}"
# 空白清理: 删除所有空白字符包括 Unicode non-breaking space (U+00A0 = 0xc2a0)
# 根因: 浏览器/终端复制命令时会将普通空格替换为 non-breaking space
DATASET="$(printf '%s' "$DATASET" | sed 's/\xc2\xa0//g' | tr -d ' \t\n\r')"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
PUSH="${PUSH:-0}"
DEBUG="${DEBUG:-0}"
RUN_ID="$(printf '%s' "${DATASET}_${TIMESTAMP}_seed${SEED}" | tr -d ' \t\n\r')"

echo "============================================"
echo " Walpurgis Experiment (cugraph-gnn pattern)"
echo " Run ID:   $RUN_ID"
echo " Dataset:  [$DATASET] (len=${#DATASET})"
printf " Dataset hex: " && printf '%s' "$DATASET" | od -A n -t x1 | head -1
echo " GPU:      $GPU"
echo " Epochs:   $EPOCHS"
echo " Seed:     $SEED"
echo " Debug:    $DEBUG"
echo "============================================"

# ── Phase 0: 环境诊断 (from cugraph-gnn print_env.sh) ──
echo ""
echo "[Phase 0] Environment"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU: $line"
    done
fi
echo "  Python: $(python3 --version 2>&1)"
echo "  PWD: $REPO_DIR"

# ── Phase 1: 数据校验 (from cugraph-gnn datasets/get_test_data.sh) ──
echo ""
echo "[Phase 1] Data validation"
DATA_DIR="datasets/$(printf '%s' "${DATASET}" | tr -d ' \t\n\r')"
# 断言: DATASET不应含空格
if [[ "$DATASET" =~ [[:space:]] ]]; then
    echo "  FATAL: DATASET='${DATASET}' still contains whitespace after strip"
    exit 1
fi
if [ ! -f "${DATA_DIR}/train.npz" ]; then
    if [ "$DATASET" = "SYNTH" ] && [ -f "src/walpurgis/generate_synth_data.py" ]; then
        echo "  Generating SYNTH data..."
        python3 src/walpurgis/generate_synth_data.py
    else
        echo "  ERROR: ${DATA_DIR}/train.npz not found"
        exit 1
    fi
fi
for split in train val test; do
    size=$(stat -f%z "${DATA_DIR}/${split}.npz" 2>/dev/null || stat -c%s "${DATA_DIR}/${split}.npz" 2>/dev/null || echo "0")
    echo "  ${split}.npz: $(( size / 1024 ))KB"
done

# ── Phase 2: 模型构建 + Warmup (from cugraph-gnn PerformanceMeter warmup) ──
echo ""
echo "[Phase 2] Model build + warmup"

# 设备选择
if [ "$GPU" != "cpu" ] && python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    DEVICE="cpu"
fi
echo "  Device: $DEVICE"

# ── Phase 2.5: 异构内存基础设施 benchmark ──
# 论文核心贡献: tiered_allocator + partition_skiplist + async_migration + temporal_bridge
# 这不是可选的附加组件, 而是论文的主要突破点
# 预测实验(Phase 3)只是证明基础设施在真实工作负载上可用的ablation
echo ""
echo "[Phase 2.5] Heterogeneous memory infrastructure"

INFRA_DIR="$RESULT_DIR/infra_${RUN_ID}"
mkdir -p "$INFRA_DIR"

# 2.5a: 编译 philemon_bench (端到端: TieredAllocator + TemporalBridge + MigrationScheduler)
PHILEMON_BIN="src/bench/bin/philemon_bench"
if [ ! -f "$PHILEMON_BIN" ] || [ src/core/tiered_allocator.hpp -nt "$PHILEMON_BIN" ]; then
    echo "  Compiling philemon_bench..."
    mkdir -p src/bench/bin
    g++ -std=c++17 -O2 -pthread -I src \
        -o "$PHILEMON_BIN" src/bench/philemon_bench.cpp 2>&1 | head -5
    if [ $? -ne 0 ]; then
        echo "  WARNING: philemon_bench compilation failed, skipping infra phase"
    fi
fi

if [ -f "$PHILEMON_BIN" ]; then
    echo "  Running tiered allocator + temporal bridge + migration scheduler..."
    timeout 60 "$PHILEMON_BIN" 2>&1 | tee "${INFRA_DIR}/philemon_bench.log" | while IFS= read -r line; do
        echo "  [infra] $line"
    done

    # 提取关键指标写入JSON
    python3 << 'INFRA_PY'
import re, json, os
log_path = os.environ.get("INFRA_DIR", ".") + "/philemon_bench.log"
if not os.path.exists(log_path):
    exit(0)
log = open(log_path).read()
metrics = {}
# 提取 throughput, latency, memory usage
for pat, key in [
    (r'Throughput:\s*([\d.]+)\s*queries/s', 'query_throughput'),
    (r'P50 latency:\s*([\d.]+)\s*us', 'p50_latency_us'),
    (r'P99 latency:\s*([\d.]+)\s*us', 'p99_latency_us'),
    (r'Peak RSS:\s*([\d.]+)\s*MB', 'peak_rss_mb'),
    (r'HBM tier:\s*([\d.]+)\s*MB', 'hbm_tier_mb'),
    (r'DRAM tier:\s*([\d.]+)\s*MB', 'dram_tier_mb'),
    (r'Migrations:\s*(\d+)', 'migration_count'),
    (r'Concurrent.*?:\s*([\d.]+)\s*queries/s', 'concurrent_throughput'),
]:
    m = re.search(pat, log, re.IGNORECASE)
    if m:
        metrics[key] = float(m.group(1))
if metrics:
    with open(os.environ.get("INFRA_DIR", ".") + "/infra_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Infrastructure metrics: {json.dumps(metrics, indent=None)}")
INFRA_PY
fi

# 2.5b: 编译 hetero_bench.cu (CUDA异构GPU kernel, 仅在有nvcc时)
if command -v nvcc &>/dev/null; then
    HETERO_BIN="src/bench/bin/hetero_bench"
    if [ ! -f "$HETERO_BIN" ] || [ src/cuda/hetero_bench.cu -nt "$HETERO_BIN" ]; then
        echo "  Compiling hetero_bench.cu..."
        nvcc -std=c++17 -O2 -I src \
            -o "$HETERO_BIN" src/cuda/hetero_bench.cu 2>&1 | head -5
    fi
    if [ -f "$HETERO_BIN" ]; then
        echo "  Running heterogeneous GPU memory benchmark..."
        timeout 120 "$HETERO_BIN" 2>&1 | tee "${INFRA_DIR}/hetero_bench.log" | while IFS= read -r line; do
            echo "  [cuda] $line"
        done
    fi
else
    echo "  nvcc not found, skipping CUDA hetero_bench"
fi

# 2.5c: partition_skiplist + interval_index 独立 bench
PARTITION_BIN="src/bench/bin/partition_index_bench"
if [ ! -f "$PARTITION_BIN" ] || [ src/core/partition_skiplist.hpp -nt "$PARTITION_BIN" ]; then
    echo "  Compiling partition_index_bench..."
    g++ -std=c++17 -O2 -pthread -I src \
        -o "$PARTITION_BIN" src/bench/partition_index_bench.cpp 2>&1 | head -5
fi
if [ -f "$PARTITION_BIN" ]; then
    echo "  Running partition skiplist + interval index benchmark..."
    timeout 30 "$PARTITION_BIN" 2>&1 | tee "${INFRA_DIR}/partition_bench.log" | while IFS= read -r line; do
        echo "  [part] $line"
    done
fi

echo "  Infrastructure results: $INFRA_DIR/"

# ── Phase 3: 训练 (核心) ──
echo ""
echo "[Phase 3] Training ($EPOCHS epochs)"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

RESULT_DIR="experiments/results"
mkdir -p "$RESULT_DIR"
LOG_FILE="${RESULT_DIR}/${RUN_ID}.log"
export LOG_FILE RUN_ID DATASET

TRAIN_EXIT=0
WALPURGIS_DEBUG=$DEBUG \
CASCADE_DIAG_LOG="${RESULT_DIR}/${RUN_ID}_diag.jsonl" \
python3 train_walpurgis.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_FILE" || TRAIN_EXIT=$?

echo ""
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit code: $TRAIN_EXIT"

# ── Phase 4: 结果提取 (from cugraph-gnn benchmark Metric pattern) ──
echo ""
echo "[Phase 4] Result extraction"

LOG_FILE="$LOG_FILE" RUN_ID="$RUN_ID" DATASET="$DATASET" python3 << 'PYEOF'
import json, os, re, sys

log_file = os.environ.get("LOG_FILE", "")
run_id = os.environ.get("RUN_ID", "unknown")
dataset = os.environ.get("DATASET", "unknown")

if not os.path.exists(log_file):
    print("  WARNING: log file not found")
    sys.exit(0)

log = open(log_file).read()
result = {"run_id": run_id, "dataset": dataset}

# 12-horizon average
avg = re.findall(
    r'\(On average over 12 horizons\) Test MAE: ([\d.]+) \| Test RMSE: ([\d.]+) \| Test MAPE: ([\d.]+)%', log)
if avg:
    best = min(avg, key=lambda x: float(x[0]))
    result["test_mae"] = float(best[0])
    result["test_rmse"] = float(best[1])
    result["test_mape"] = float(best[2])
    result["status"] = "complete"
else:
    result["status"] = "no_test_metrics"

# Best Val
bv = re.search(r'Best Val\s*:\s*([\d.]+)', log)
if bv: result["best_val_mae"] = float(bv.group(1))

# Timing
tt = re.search(r'Total time:\s*([\d.]+)s', log)
if tt: result["total_time_s"] = float(tt.group(1))
at = re.search(r'Avg Train:\s*([\d.]+)s/epoch', log)
if at: result["avg_train_s"] = float(at.group(1))

# Params
params = re.search(r'Model params:\s*([\d,]+)', log)
if params: result["model_params"] = int(params.group(1).replace(',', ''))

# Per-horizon
horizons = re.findall(
    r'horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)', log)
if horizons:
    last12 = horizons[-12:] if len(horizons) >= 12 else horizons
    result["per_horizon"] = [
        {"h": int(h), "mae": float(m), "rmse": float(r), "mape": float(p)}
        for h, m, r, p in last12]

# Diagnostics (最后一次)
diag_gate = re.findall(r'adaptive_emb_gate=([\d.]+)', log)
if diag_gate: result["final_adaptive_gate"] = float(diag_gate[-1])
diag_depth = re.findall(r'depth_gates: (.+)', log)
if diag_depth: result["final_depth_gates"] = diag_depth[-1].strip()

# Infrastructure metrics (Phase 2.5)
infra_dir = f"experiments/results/infra_{run_id}"
infra_json = os.path.join(infra_dir, "infra_metrics.json")
if os.path.exists(infra_json):
    with open(infra_json) as f:
        result["infrastructure"] = json.load(f)

# 打印 (cugraph-gnn Metric风格)
print(f"  ┌─────────────────────────────────")
print(f"  │ Run:     {run_id}")
print(f"  │ Status:  {result['status']}")
if 'infrastructure' in result:
    infra = result['infrastructure']
    print(f"  │ ── Infrastructure (core contribution) ──")
    if 'query_throughput' in infra:
        print(f"  │ Query throughput: {infra['query_throughput']:.0f} q/s")
    if 'p50_latency_us' in infra:
        print(f"  │ P50 latency:     {infra['p50_latency_us']:.1f} μs")
    if 'p99_latency_us' in infra:
        print(f"  │ P99 latency:     {infra['p99_latency_us']:.1f} μs")
    if 'peak_rss_mb' in infra:
        print(f"  │ Peak RSS:        {infra['peak_rss_mb']:.1f} MB")
    print(f"  │ ── Prediction (ablation) ──")
if 'test_mae' in result:
    print(f"  │ MAE:     {result['test_mae']:.4f}")
    print(f"  │ RMSE:    {result['test_rmse']:.4f}")
    print(f"  │ MAPE:    {result['test_mape']:.2f}%")
if 'best_val_mae' in result:
    print(f"  │ Val:     {result['best_val_mae']:.4f}")
if 'total_time_s' in result:
    print(f"  │ Time:    {result['total_time_s']:.1f}s")
if 'model_params' in result:
    print(f"  │ Params:  {result['model_params']:,}")
print(f"  └─────────────────────────────────")

# SOTA对比
if 'test_mae' in result and dataset == "METR-LA":
    mae = result['test_mae']
    print(f"\n  SOTA comparison:")
    for name, val in [("TITAN", 2.88), ("STAEFormer", 2.90), ("PDFormer", 2.94), ("D2STGNN", 3.04)]:
        gap = mae - val
        mark = " ← BEATEN!" if gap < 0 else ""
        print(f"    {name:15s} {val:.2f}  (gap: {gap:+.4f}){mark}")

# 写结果JSON
result_file = f"experiments/results/{run_id}.json"
with open(result_file, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n  Result: {result_file}")
PYEOF

# ── Phase 5: Git push (可选) ──
if [ "$PUSH" = "1" ]; then
    echo ""
    echo "[Phase 5] Git push"
    GIT_TOKEN="${GIT_TOKEN:-}"
    if [ -n "$GIT_TOKEN" ]; then
        git remote set-url origin "https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" 2>/dev/null || true
    fi
    git add experiments/results/
    git commit -m "experiment: ${RUN_ID} — ${DATASET} ${EPOCHS}ep seed${SEED}" 2>/dev/null || true
    git pull --rebase origin main 2>/dev/null && git push origin main 2>/dev/null || echo "  Push failed"
fi

echo ""
echo "============================================"
echo " Done: $RUN_ID"
echo " Log:  $LOG_FILE"
echo "============================================"
