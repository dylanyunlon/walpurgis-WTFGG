#!/usr/bin/env bash
# experiments/run_baselines.sh — 下载并运行 D2STGNN/STAEFormer 基线
# 用法: bash experiments/run_baselines.sh [--gpu 0] [--model d2stgnn|staeformer|all]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

GPU="${GPU:-1}"
MODEL="${MODEL:-all}"
DATASET="${DATASET:-METR-LA}"
BASELINES_DIR="experiments/baselines"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";     shift 2;;
        --model)   MODEL="$2";   shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        *) shift;;
    esac
done

set +u
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate walpurgis 2>/dev/null || true
set -u

mkdir -p "$BASELINES_DIR"

# ── D2STGNN 基线 (upstream 代码) ────────────────────────
run_d2stgnn() {
    echo "============================================"
    echo " D2STGNN Baseline on $DATASET"
    echo "============================================"
    local BDIR="$BASELINES_DIR/d2stgnn"
    mkdir -p "$BDIR"

    # 复制 upstream 代码到工作区
    if [ ! -f "$BDIR/main.py" ]; then
        cp -r upstream/d2stgnn/* "$BDIR/"
    fi

    # 确保数据链接
    ln -sf "$REPO_DIR/datasets" "$BDIR/datasets" 2>/dev/null || true

    # 修改 config 的 epochs 和 device
    cd "$BDIR"
    python3 -c "
import yaml
cfg_path = 'configs/${DATASET}.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg['start_up']['device'] = 'cuda:0'
cfg['optim_args']['epochs'] = 100
cfg['optim_args']['patience'] = 15
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f'Config updated: epochs=100, device=cuda:0')
"

    CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --dataset "$DATASET" \
        2>&1 | tee "$BDIR/train_${DATASET}.log"
    cd "$REPO_DIR"
}

# ── STAEFormer 基线 ─────────────────────────────────────
run_staeformer() {
    echo "============================================"
    echo " STAEFormer Baseline on $DATASET"
    echo "============================================"
    local BDIR="$BASELINES_DIR/staeformer"

    if [ ! -d "$BDIR" ]; then
        echo "Cloning STAEFormer..."
        git clone https://github.com/XDZhelheim/STAEformer.git "$BDIR"
    fi

    cd "$BDIR"
    # STAEFormer 需要特定数据格式, 链接数据
    ln -sf "$REPO_DIR/datasets" "$BDIR/data" 2>/dev/null || true

    echo "STAEFormer training — see $BDIR/README.md for data prep"
    echo "典型命令:"
    echo "  CUDA_VISIBLE_DEVICES=$GPU python3 train.py --dataset $DATASET --epochs 100"
    cd "$REPO_DIR"
}

# ── 执行 ────────────────────────────────────────────────
case "$MODEL" in
    d2stgnn)    run_d2stgnn ;;
    staeformer) run_staeformer ;;
    all)        run_d2stgnn; run_staeformer ;;
    *) echo "Unknown model: $MODEL (use d2stgnn|staeformer|all)"; exit 1;;
esac

echo ""
echo "Baselines complete. Results in $BASELINES_DIR/"
