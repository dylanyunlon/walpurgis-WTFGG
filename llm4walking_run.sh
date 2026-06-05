#!/bin/bash
# ============================================================
# LLM4Walking — Walpurgis Training Pipeline
# D2STGNN 移植版, 算法改动 ≥20%, 全链路断点调试
# Upstream: D2STGNN (VLDB 2022)
# ============================================================
set -e

echo "=========================================="
echo "   LLM4Walking / Walpurgis Pipeline"
echo "   Spatio-Temporal Graph Neural Network"
echo "=========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ========== 配置 ==========
PROJECT_DIR="${SCRIPT_DIR}/src/walpurgis_walking"
DATASET="${DATASET:-METR-LA}"
DEVICE="${DEVICE:-cpu}"             # 你的机器上改成 cuda:0
EPOCHS="${EPOCHS:-5}"
DEBUG="${WALPURGIS_DEBUG:-1}"
CONDA_ENV_NAME="llm4walking"
BASE_LR="${BASE_LR:-0.002}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"
export WALPURGIS_DEBUG="${DEBUG}"

# ========== 工具 ==========
print_step() { echo ""; echo "=== $1 ==="; echo ""; }
timestamp() { date +%Y%m%d_%H%M%S; }

# ========== 环境检查 ==========
check_system() {
    print_step "Step 0: 系统环境检查"

    echo "Python: $(python3 --version 2>&1)"
    python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')
else:
    print(f'  Running on CPU')
"
    echo "Debug: WALPURGIS_DEBUG=${DEBUG}"
    echo "✓ 环境就绪"
}

# ========== Conda 环境 ==========
setup_environment() {
    print_step "Step 0.5: Conda 环境 (${CONDA_ENV_NAME})"

    if ! command -v conda &>/dev/null; then
        echo "⚠ Conda 不可用, 跳过环境创建, 使用当前 Python"
        return 0
    fi

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "环境已存在, 激活..."
        eval "$(conda shell.bash hook)"
        conda activate ${CONDA_ENV_NAME}
        return 0
    fi

    echo "创建 conda env: ${CONDA_ENV_NAME} (Python 3.10)..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}

    pip install --upgrade pip
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install pyyaml numpy scipy pandas matplotlib
    pip install tensorboard wandb

    echo "✓ 环境就绪"
}

# ========== 数据生成 ==========
generate_data() {
    print_step "Step 1: 数据生成"
    if [ -f "${PROJECT_DIR}/datasets/METR-LA/train.npz" ]; then
        echo "数据已存在, 跳过"
        return 0
    fi
    python3 generate_synth_data.py
    echo "✓ 数据生成完毕"
}

# ========== 数据检查 (断点调试风格) ==========
inspect_data() {
    print_step "Step 2: 数据断点检查"
    python3 << 'PYEOF'
import numpy as np, pickle, os
print("=" * 60)
print("[INSPECT] 数据完整性")
print("=" * 60)
base = "walpurgis_walking/datasets"
for sp in ["train", "val", "test"]:
    p = os.path.join(base, "METR-LA", f"{sp}.npz")
    if not os.path.exists(p):
        print(f"  ✗ MISSING: {p}"); continue
    d = np.load(p)
    x, y = d['x'], d['y']
    print(f"\n  [{sp}] x={x.shape} y={y.shape}")
    print(f"    speed: min={x[...,0].min():.2f} max={x[...,0].max():.2f} mean={x[...,0].mean():.2f}")
    print(f"    NaN={np.isnan(x).sum()} Inf={np.isinf(x).sum()}")

ap = os.path.join(base, "sensor_graph", "adj_mx_la.pkl")
if os.path.exists(ap):
    with open(ap,'rb') as f:
        ids,_,adj = pickle.load(f)
    print(f"\n  [adj] shape={adj.shape} sensors={len(ids)}")
    print(f"    range=[{adj.min():.4f}, {adj.max():.4f}]")
    print(f"    sparsity={(adj==0).sum()/adj.size:.1%}")
print("\n" + "=" * 60)
PYEOF
    echo "✓ 检查完毕"
}

# ========== 模型结构检查 ==========
inspect_model() {
    print_step "Step 3: 模型结构 + 参数快照"
    cd "${PROJECT_DIR}"
    python3 << 'PYEOF'
import os, yaml, torch, pickle
os.environ["WALPURGIS_DEBUG"] = "0"
from models.model import D2STGNN
from walpurgis_walking import snapshot_model
from utils.cal_adj import transition_matrix

with open("configs/METR-LA.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
with open(cfg['data_args']['adj_data_path'], 'rb') as f:
    _,_,adj = pickle.load(f)
N = adj.shape[0]
ma = cfg['model_args']
ma['device'] = torch.device('cpu')
ma['num_nodes'] = N
at = [transition_matrix(adj).T, transition_matrix(adj.T).T]
ma['adjs'] = [torch.tensor(a,dtype=torch.float32) for a in at]
ma['adjs_ori'] = torch.tensor(adj,dtype=torch.float32)
ma['dataset'] = 'METR-LA'

model = D2STGNN(**ma)
total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {train:,}  Size: {total*4/1e6:.2f}MB")
for nm, m in model.named_children():
    p = sum(pp.numel() for pp in m.parameters())
    print(f"  {nm:30s} {p:>8,} params")
snapshot_model(model, epoch=0, step=0, top_k=5)
PYEOF
    cd "${SCRIPT_DIR}"
    echo "✓ 模型检查完毕"
}

# ========== Smoke Test ==========
smoke_test() {
    print_step "Step 4: Smoke Test (1轮 fwd+bwd)"
    cd "${PROJECT_DIR}"
    WALPURGIS_DEBUG=1 python3 << 'PYEOF'
import os, yaml, torch, pickle
os.environ["WALPURGIS_DEBUG"] = "1"
from models.model import D2STGNN
from models.losses import masked_mae
from walpurgis_walking import (snapshot_model, register_activation_hooks,
                               gradient_health_check)
from utils.cal_adj import transition_matrix

with open("configs/METR-LA.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
with open(cfg['data_args']['adj_data_path'], 'rb') as f:
    _,_,adj = pickle.load(f)
N = adj.shape[0]
ma = cfg['model_args']
ma['device'] = torch.device('cpu')
ma['num_nodes'] = N
at = [transition_matrix(adj).T, transition_matrix(adj.T).T]
ma['adjs'] = [torch.tensor(a,dtype=torch.float32) for a in at]
ma['adjs_ori'] = torch.tensor(adj,dtype=torch.float32)
ma['dataset'] = 'METR-LA'

model = D2STGNN(**ma)
opt = torch.optim.Adam(model.parameters(), lr=0.002)
tracker = register_activation_hooks(model)

B, T = 4, 12
x = torch.randn(B,T,N,3)
x[...,0] = x[...,0].abs()*30+20
x[...,1] = torch.rand(B,T,N)*0.99
x[...,2] = torch.randint(0,7,(B,T,N)).float()
y = x[...,0].clone()

model.train()
out = model(x)
pred = out.transpose(1,2)
loss = masked_mae(pred, y[:,:pred.shape[1],:], 0.0)
print(f"\nLoss={loss.item():.4f}")

opt.zero_grad()
loss.backward()
gradient_health_check(model)
tracker.report()
tracker.check_dead()
tracker.remove()
snapshot_model(model, epoch=0, step=1, top_k=5)
opt.step()

model.eval()
with torch.no_grad():
    o2 = model(x).transpose(1,2)
    l2 = masked_mae(o2, y[:,:o2.shape[1],:], 0.0)
print(f"After step: {l2.item():.4f}  delta={loss.item()-l2.item():.6f}")
print("✓ Smoke test passed")
PYEOF
    cd "${SCRIPT_DIR}"
}

# ========== 训练 ==========
run_training() {
    print_step "Step 5: 训练 (WALPURGIS_DEBUG=${DEBUG})"
    cd "${PROJECT_DIR}"
    mkdir -p output/logs
    TS=$(timestamp)
    LOG="output/logs/train_${TS}.log"
    echo "Dataset: ${DATASET}  Device: ${DEVICE}  Log: ${LOG}"
    python3 main.py --dataset ${DATASET} 2>&1 | tee "${LOG}"
    EXIT=$?
    cd "${SCRIPT_DIR}"
    [ $EXIT -eq 0 ] && echo "✓ 训练完成" || echo "✗ 训练失败 ($EXIT)"
    return $EXIT
}

# ========== 帮助 ==========
show_help() {
    cat << 'EOF'
LLM4Walking — Walpurgis Pipeline

Usage: ./llm4walking_run.sh [command]

Commands:
  setup        Conda 环境创建
  check        Python/GPU 检查
  data         生成合成 METR-LA 数据
  inspect      数据断点检查
  model        模型结构 + 参数快照
  smoke        一轮 forward/backward + 梯度/激活诊断
  train        完整训练
  all          全链路 (check→data→inspect→model→smoke)

Env Vars:
  WALPURGIS_DEBUG=1          全量调试
  WALPURGIS_DEBUG=model,trainer  按模块
  DATASET=METR-LA            配置名
  DEVICE=cuda:0              GPU
  EPOCHS=5                   轮数
EOF
}

# ========== 主入口 ==========
main() {
    CMD=${1:-"help"}
    case $CMD in
        setup)      setup_environment ;;
        check)      check_system ;;
        data)       generate_data ;;
        inspect)    inspect_data ;;
        model)      inspect_model ;;
        smoke)      smoke_test ;;
        train)      run_training ;;
        all)
            check_system
            generate_data
            inspect_data
            inspect_model
            smoke_test
            echo ""
            echo "全部检查通过. 执行训练:"
            echo "  WALPURGIS_DEBUG=1 ./llm4walking_run.sh train"
            ;;
        help|--help|-h) show_help ;;
        *) echo "Unknown: $CMD"; show_help; exit 1 ;;
    esac
}
main "$@"
