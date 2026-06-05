#!/bin/bash
# ============================================================
# LLM4Nightfall — Walpurgis Training Pipeline
# D2STGNN 移植版 (Nightfall变体), 算法改动 ≥20%, 全链路断点调试
# Upstream: D2STGNN (VLDB 2022)
# 算法改写: spectral gating + adaptive温度 + 梯度噪声退火
# ============================================================
set -e

echo "=========================================="
echo "   LLM4Nightfall / Walpurgis Pipeline"
echo "   Spatio-Temporal Graph Neural Network"
echo "   Nightfall Variant (AdamW + CosineAnnealing)"
echo "=========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# ========== 配置 ==========
PROJECT_DIR="${SCRIPT_DIR}/src/walpurgis_nightfall"
DATASET="${DATASET:-SYNTH}"
DEVICE="${DEVICE:-cpu}"             # 有GPU的机器改成 cuda:0
EPOCHS="${EPOCHS:-3}"
DEBUG="${NIGHTFALL_DEBUG:-0}"
CONDA_ENV_NAME="llm4nightfall"
BASE_LR="${BASE_LR:-0.002}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATA_ROOT="${SCRIPT_DIR}/datasets"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"
export NIGHTFALL_DEBUG="${DEBUG}"

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
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print(f'  Running on CPU')
import numpy as np
print(f'  NumPy:   {np.__version__}')
"
    echo "Debug: NIGHTFALL_DEBUG=${DEBUG}"
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
    pip install pyyaml numpy scipy pandas matplotlib scikit-learn
    pip install tensorboard

    echo "✓ 环境就绪"
}

# ========== pytest冒烟测试 ==========
run_pytest() {
    print_step "Step 0.8: pytest 冒烟测试 (15 tests)"
    PYTHONPATH="${SCRIPT_DIR}/src" python3 -m pytest \
        "${SCRIPT_DIR}/tests/test_nightfall_smoke.py" -v --tb=short
    echo "✓ pytest 全部通过"
}

# ========== 数据生成 ==========
generate_data() {
    print_step "Step 1: 合成数据生成"
    if [ -f "${DATA_ROOT}/SYNTH/train.npz" ]; then
        echo "SYNTH数据已存在, 跳过生成"
        python3 -c "
import numpy as np
d = np.load('${DATA_ROOT}/SYNTH/train.npz')
print(f'  [已存在] train: x={d[\"x\"].shape} y={d[\"y\"].shape}')
"
        return 0
    fi
    PYTHONPATH="${SCRIPT_DIR}/src" python3 -m walpurgis_nightfall.generate_synth_data
    echo "✓ 数据生成完毕"
}

# ========== 数据检查 ==========
inspect_data() {
    print_step "Step 2: 数据断点检查"
    python3 << PYEOF
import numpy as np, pickle, os
print("=" * 60)
print("[INSPECT] Nightfall数据完整性")
print("=" * 60)
base = "${DATA_ROOT}/SYNTH"
for sp in ["train", "val", "test"]:
    p = os.path.join(base, f"{sp}.npz")
    if not os.path.exists(p):
        print(f"  ✗ MISSING: {p}"); continue
    d = np.load(p)
    x, y = d['x'], d['y']
    print(f"\n  [{sp}] x={x.shape} y={y.shape}")
    print(f"    speed: min={x[...,0].min():.2f} max={x[...,0].max():.2f} mean={x[...,0].mean():.2f}")
    print(f"    NaN={np.isnan(x).sum()} Inf={np.isinf(x).sum()}")

ap = "${DATA_ROOT}/sensor_graph/adj_mx_synth.pkl"
if os.path.exists(ap):
    with open(ap,'rb') as f:
        adj = pickle.load(f)
    print(f"\n  [adj] shape={adj.shape}")
    print(f"    range=[{adj.min():.4f}, {adj.max():.4f}]")
    print(f"    sparsity={(adj==0).sum()/adj.size:.1%}")
print("\n" + "=" * 60)
PYEOF
    echo "✓ 检查完毕"
}

# ========== 模型结构检查 ==========
inspect_model() {
    print_step "Step 3: 模型结构 + 参数快照"
    NIGHTFALL_DEBUG=1 python3 << PYEOF
import os, torch, pickle
import numpy as np
os.environ["NIGHTFALL_DEBUG"] = "1"
import sys; sys.path.insert(0, "${SCRIPT_DIR}/src")
from walpurgis_nightfall.models.model import D2STGNN
from walpurgis_nightfall import snapshot_model
from walpurgis_nightfall.utils.load_data import load_adj

adj_mx, adj_ori = load_adj("${DATA_ROOT}/sensor_graph/adj_mx_synth.pkl", "doubletransition")
N = adj_mx[0].shape[0]
device = torch.device("cpu")
model_args = dict(
    batch_size=16, num_feat=1, num_hidden=16, node_hidden=8,
    time_emb_dim=8, dropout=0.1, seq_length=12, k_t=3, k_s=2,
    gap=3, num_modalities=2, device=device, num_nodes=N,
    adjs=[torch.tensor(a).to(device) for a in adj_mx],
    adjs_ori=torch.tensor(adj_ori).to(device),
    dataset="SYNTH"
)
model = D2STGNN(**model_args)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}  Size: {total*4/1e6:.2f}MB")
for nm, m in model.named_children():
    p = sum(pp.numel() for pp in m.parameters())
    print(f"  {nm:35s} {p:>8,} params")
snapshot_model(model, epoch=0, step=0, top_k=5)
PYEOF
    echo "✓ 模型检查完毕"
}

# ========== Smoke Test ==========
smoke_test() {
    print_step "Step 4: Smoke Test (1轮 fwd+bwd + 梯度/激活诊断)"
    NIGHTFALL_DEBUG=1 python3 << PYEOF
import os, torch, pickle
import numpy as np
os.environ["NIGHTFALL_DEBUG"] = "1"
import sys; sys.path.insert(0, "${SCRIPT_DIR}/src")
from walpurgis_nightfall.models.model import D2STGNN
from walpurgis_nightfall.models.losses import masked_mae
from walpurgis_nightfall import snapshot_model, register_activation_hooks, gradient_health_check
from walpurgis_nightfall.utils.load_data import load_adj

adj_mx, adj_ori = load_adj("${DATA_ROOT}/sensor_graph/adj_mx_synth.pkl", "doubletransition")
N = adj_mx[0].shape[0]
device = torch.device("cpu")
model_args = dict(
    batch_size=16, num_feat=1, num_hidden=16, node_hidden=8,
    time_emb_dim=8, dropout=0.1, seq_length=12, k_t=3, k_s=2,
    gap=3, num_modalities=2, device=device, num_nodes=N,
    adjs=[torch.tensor(a).to(device) for a in adj_mx],
    adjs_ori=torch.tensor(adj_ori).to(device),
    dataset="SYNTH"
)
model = D2STGNN(**model_args)
opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
tracker = register_activation_hooks(model)

B, T = 4, 12
x = torch.randn(B, T, N, 3)
x[..., 0] = x[..., 0].abs() * 30 + 20
x[..., 1] = torch.rand(B, T, N) * 0.99
x[..., 2] = torch.randint(0, 7, (B, T, N)).float()
y = x[..., 0].clone()

model.train()
out = model(x)
pred = out.transpose(1, 2)
loss = masked_mae(pred, y[:, :pred.shape[1], :], 0.0)
print(f"\nForward Loss={loss.item():.4f}")

opt.zero_grad()
loss.backward()
issues = gradient_health_check(model)
if not issues:
    print("  ✓ 梯度健康 (无爆炸/消失/NaN)")
tracker.report()
dead = tracker.check_dead()
if dead:
    for nm, frac in dead:
        print(f"  ⚠ Dead layer: {nm} ({frac:.1%})")
else:
    print("  ✓ 无dead神经元")
tracker.remove()
snapshot_model(model, epoch=0, step=1, top_k=5)
opt.step()

model.eval()
with torch.no_grad():
    o2 = model(x).transpose(1, 2)
    l2 = masked_mae(o2, y[:, :o2.shape[1], :], 0.0)
print(f"After step: {l2.item():.4f}  delta={loss.item()-l2.item():.6f}")
print("✓ Smoke test passed")
PYEOF
    echo "✓ Smoke test完毕"
}

# ========== 训练 ==========
run_training() {
    print_step "Step 5: 完整训练 (NIGHTFALL_DEBUG=${DEBUG}, EPOCHS=${EPOCHS})"

    mkdir -p "${PROJECT_DIR}/output/logs"
    TS=$(timestamp)
    LOG="${PROJECT_DIR}/output/logs/train_${DATASET}_${TS}.log"
    echo "Dataset: ${DATASET}  Device: ${DEVICE}  Epochs: ${EPOCHS}  Log: ${LOG}"

    python3 << PYEOF 2>&1 | tee "${LOG}"
import os, sys, time, yaml, torch
import numpy as np
os.environ["NIGHTFALL_DEBUG"] = "${DEBUG}"
sys.path.insert(0, "${SCRIPT_DIR}/src")

from walpurgis_nightfall.utils.train import set_config, EarlyStopping, data_reshaper
from walpurgis_nightfall.utils.load_data import load_dataset, load_adj
from walpurgis_nightfall.utils.log import TrainLogger
from walpurgis_nightfall.models.trainer import trainer
from walpurgis_nightfall.models.model import D2STGNN
from walpurgis_nightfall import _is_debug, snapshot_model, register_activation_hooks

set_config(0)

DATASET = "${DATASET}"
DEVICE  = torch.device("${DEVICE}")
EPOCHS  = int("${EPOCHS}")

# 数据路径映射
DATA_PATHS = {
    "SYNTH":    ("${DATA_ROOT}/SYNTH",    "${DATA_ROOT}/sensor_graph/adj_mx_synth.pkl",  "doubletransition"),
    "METR-LA":  ("${DATA_ROOT}/METR-LA",  "${DATA_ROOT}/sensor_graph/adj_mx_la.pkl",     "doubletransition"),
    "PEMS-BAY": ("${DATA_ROOT}/PEMS-BAY", "${DATA_ROOT}/sensor_graph/adj_mx_bay.pkl",    "doubletransition"),
    "PEMS04":   ("${DATA_ROOT}/PEMS04",   "${DATA_ROOT}/sensor_graph/adj_PEMS04.pkl",    "doubletransition"),
    "PEMS08":   ("${DATA_ROOT}/PEMS08",   "${DATA_ROOT}/sensor_graph/adj_PEMS08.pkl",    "doubletransition"),
}

# 模型配置
MODEL_CFGS = {
    "SYNTH":    dict(batch_size=16, num_feat=1, num_hidden=16, node_hidden=8,  time_emb_dim=8,  dropout=0.1, seq_length=12, k_t=3, k_s=2, gap=3, num_modalities=2),
    "METR-LA":  dict(batch_size=32, num_feat=1, num_hidden=32, node_hidden=10, time_emb_dim=10, dropout=0.1, seq_length=12, k_t=3, k_s=2, gap=3, num_modalities=2),
    "PEMS-BAY": dict(batch_size=32, num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12, dropout=0.1, seq_length=12, k_t=3, k_s=2, gap=3, num_modalities=2),
    "PEMS04":   dict(batch_size=32, num_feat=1, num_hidden=32, node_hidden=10, time_emb_dim=10, dropout=0.1, seq_length=12, k_t=3, k_s=2, gap=3, num_modalities=2),
    "PEMS08":   dict(batch_size=32, num_feat=1, num_hidden=32, node_hidden=10, time_emb_dim=10, dropout=0.1, seq_length=12, k_t=3, k_s=2, gap=3, num_modalities=2),
}

# 优化器配置
OPTIM_CFGS = {
    "SYNTH":    dict(lrate=float("${BASE_LR}"), wdecay=1e-5, eps=1e-8, lr_schedule=True, lr_sche_steps=[1,5,8], lr_decay_ratio=0.5, if_cl=True, cl_epochs=2, warm_epochs=0, output_seq_len=12, patience=100, epochs=EPOCHS, seq_length=12),
    "METR-LA":  dict(lrate=float("${BASE_LR}"), wdecay=1e-5, eps=1e-8, lr_schedule=True, lr_sche_steps=[1,30,38,46,54,62,70,80], lr_decay_ratio=0.5, if_cl=True, cl_epochs=6, warm_epochs=0, output_seq_len=12, patience=100, epochs=EPOCHS, seq_length=12),
    "PEMS-BAY": dict(lrate=float("${BASE_LR}"), wdecay=1e-5, eps=1e-8, lr_schedule=True, lr_sche_steps=[1,30,38,46,54,62,70,80], lr_decay_ratio=0.5, if_cl=True, cl_epochs=6, warm_epochs=0, output_seq_len=12, patience=100, epochs=EPOCHS, seq_length=12),
    "PEMS04":   dict(lrate=float("${BASE_LR}"), wdecay=1e-5, eps=1e-8, lr_schedule=True, lr_sche_steps=[1,30,38,46,54,62,70,80], lr_decay_ratio=0.5, if_cl=True, cl_epochs=6, warm_epochs=0, output_seq_len=12, patience=100, epochs=EPOCHS, seq_length=12),
    "PEMS08":   dict(lrate=float("${BASE_LR}"), wdecay=1e-5, eps=1e-8, lr_schedule=True, lr_sche_steps=[1,30,38,46,54,62,70,80], lr_decay_ratio=0.5, if_cl=True, cl_epochs=6, warm_epochs=0, output_seq_len=12, patience=100, epochs=EPOCHS, seq_length=12),
}

data_dir, adj_path, adj_type = DATA_PATHS[DATASET]
model_cfg = MODEL_CFGS[DATASET]
optim_cfg = OPTIM_CFGS[DATASET]

print(f"[NF] Dataset={DATASET} Device={DEVICE} Epochs={EPOCHS}")

# 加载数据
t1 = time.time()
dataloader = load_dataset(data_dir, model_cfg["batch_size"], model_cfg["batch_size"], model_cfg["batch_size"], DATASET)
print(f"Load dataset: {time.time()-t1:.2f}s  train={len(dataloader['train_loader'])} batches")
scaler = dataloader["scaler"]

# 加载邻接矩阵
t1 = time.time()
adj_mx, adj_ori = load_adj(adj_path, adj_type)
print(f"Load adj: {time.time()-t1:.2f}s  nodes={adj_mx[0].shape[0]}")

# 构造模型参数
model_args = dict(**model_cfg)
model_args["device"]   = DEVICE
model_args["num_nodes"] = adj_mx[0].shape[0]
model_args["adjs"]     = [torch.tensor(a, dtype=torch.float32).to(DEVICE) for a in adj_mx]
model_args["adjs_ori"] = torch.tensor(adj_ori, dtype=torch.float32).to(DEVICE)
model_args["dataset"]  = DATASET

# 初始化模型
model = D2STGNN(**model_args).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params  ({total_params*4/1e6:.2f}MB)")

# 训练参数
n_batches = len(dataloader["train_loader"])
optim_cfg["cl_steps"]  = optim_cfg["cl_epochs"] * n_batches
optim_cfg["warm_steps"] = optim_cfg["warm_epochs"] * n_batches
optim_cfg["print_model"] = False

# 初始快照
if _is_debug():
    snapshot_model(model, epoch=0, step=0, top_k=5)

# 训练前 activation probe
if _is_debug():
    print("\n[NF] === First-batch activation probe ===")
    tracker = register_activation_hooks(model)
    model.train()
    for x, y in dataloader["train_loader"].get_iterator():
        probe_x = data_reshaper(x, DEVICE)
        with torch.no_grad():
            _ = model(probe_x)
        break
    tracker.report()
    tracker.remove()
    print("[NF] === Probe complete ===\n")

# 引擎 + 早停
engine = trainer(scaler, model, **optim_cfg)
os.makedirs("${PROJECT_DIR}/output", exist_ok=True)
save_path = "${PROJECT_DIR}/output/D2STGNN_" + DATASET + ".pt"
save_path_resume = "${PROJECT_DIR}/output/D2STGNN_" + DATASET + "_resume.pt"
early_stopping = EarlyStopping(optim_cfg["patience"], save_path)

train_times, val_times = [], []
batch_num = 0

for epoch in range(1, optim_cfg["epochs"] + 1):
    t_train = time.time()
    train_loss, train_mape, train_rmse = [], [], []
    dataloader["train_loader"].shuffle()
    for x, y in dataloader["train_loader"].get_iterator():
        trainx = data_reshaper(x, DEVICE)
        trainy = data_reshaper(y, DEVICE)
        mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=None, _min=None)
        train_loss.append(mae); train_mape.append(mape); train_rmse.append(rmse)
        batch_num += 1
    train_dt = time.time() - t_train
    train_times.append(train_dt)

    if engine.if_lr_scheduler:
        engine.lr_scheduler.step()

    t_val = time.time()
    mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(DEVICE, dataloader, "D2STGNN", _max=None, _min=None)
    val_dt = time.time() - t_val
    val_times.append(val_dt)

    curr_lr = engine.optimizer.param_groups[0]["lr"]
    log = (f"Epoch {epoch:03d} | "
           f"Train MAE={np.mean(train_loss):.4f} MAPE={np.mean(train_mape):.4f} RMSE={np.mean(train_rmse):.4f} | "
           f"Val MAE={mvalid_loss:.4f} RMSE={mvalid_rmse:.4f} MAPE={mvalid_mape:.4f} | "
           f"LR={curr_lr:.6f} | {train_dt:.1f}s+{val_dt:.1f}s")
    print(log)

    early_stopping(mvalid_loss, engine.model)
    if early_stopping.early_stop:
        print("Early stopping!")
        break

    engine.test(model, save_path_resume, DEVICE, dataloader, scaler, "D2STGNN",
                _max=None, _min=None, loss=engine.loss, dataset_name=DATASET)

print(f"\nAvg Train: {np.mean(train_times):.2f}s/epoch  Avg Val: {np.mean(val_times):.2f}s/epoch")
print("\n=== Final Test Metrics ===")
engine.test(model, save_path_resume, DEVICE, dataloader, scaler, "D2STGNN",
            save=False, _max=None, _min=None, loss=engine.loss, dataset_name=DATASET)
print(f"\n[NF] Training complete. Checkpoint: {save_path_resume}")
PYEOF
    EXIT=$?
    [ $EXIT -eq 0 ] && echo "✓ 训练完成" || { echo "✗ 训练失败 ($EXIT)"; return $EXIT; }
}

# ========== 帮助 ==========
show_help() {
    cat << 'EOF'
LLM4Nightfall — Walpurgis Nightfall Pipeline

Usage: ./llm4nightfall_run.sh [command]

Commands:
  setup        Conda 环境创建
  check        Python/GPU 检查
  pytest       pytest 15/15 冒烟测试
  data         生成合成 SYNTH 数据
  inspect      数据断点检查
  model        模型结构 + 参数快照
  smoke        一轮 forward/backward + 梯度/激活诊断
  train        完整训练
  all          全链路 (check→pytest→data→inspect→model→smoke→train)

Env Vars:
  NIGHTFALL_DEBUG=1          全量调试 (激活/梯度/快照)
  NIGHTFALL_DEBUG=model,trainer  按模块调试
  DATASET=SYNTH              数据集 (SYNTH/METR-LA/PEMS-BAY/PEMS04/PEMS08)
  DEVICE=cuda:0              GPU设备
  EPOCHS=3                   训练轮数
  BASE_LR=0.002              初始学习率
  BATCH_SIZE=16              批次大小

Examples:
  ./llm4nightfall_run.sh all
  DATASET=SYNTH EPOCHS=2 ./llm4nightfall_run.sh train
  NIGHTFALL_DEBUG=1 ./llm4nightfall_run.sh smoke
  DEVICE=cuda:0 DATASET=METR-LA EPOCHS=80 ./llm4nightfall_run.sh train
EOF
}

# ========== 主入口 ==========
main() {
    CMD=${1:-"help"}
    case $CMD in
        setup)      setup_environment ;;
        check)      check_system ;;
        pytest)     run_pytest ;;
        data)       generate_data ;;
        inspect)    inspect_data ;;
        model)      inspect_model ;;
        smoke)      smoke_test ;;
        train)      run_training ;;
        all)
            check_system
            run_pytest
            generate_data
            inspect_data
            inspect_model
            smoke_test
            run_training
            echo ""
            echo "=========================================="
            echo "  全链路完成 ✓"
            echo "  LLM4Nightfall pipeline 验证通过"
            echo "=========================================="
            ;;
        help|--help|-h) show_help ;;
        *) echo "Unknown command: $CMD"; show_help; exit 1 ;;
    esac
}
main "$@"
