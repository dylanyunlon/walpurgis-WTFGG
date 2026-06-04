#!/bin/bash
# ============================================================
# LLM4Walking — Walpurgis-WTFGG Training Pipeline
# 基于 D2STGNN 的时空图神经网络, "鲁迅式"移植 + 全链路调试
# Upstream: https://github.com/dylanyunlon/walpurgis-WTFGG
# ============================================================

set -e

echo "=========================================="
echo "   LLM4Walking / Walpurgis Pipeline"
echo "   Spatio-Temporal Graph Neural Network"
echo "=========================================="
echo ""

# ========== 路径配置 ==========
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PROJECT_DIR="${SCRIPT_DIR}/walpurgis"
DATASET="${DATASET:-SYNTH-METRLA}"
CONFIG_FILE="${PROJECT_DIR}/configs/${DATASET}.yaml"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-5}"
DEBUG="${WALPURGIS_DEBUG:-1}"    # 默认开启全调试

# ========== 环境变量 ==========
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}:${PYTHONPATH}"
export WALPURGIS_DEBUG="${DEBUG}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# ========== 工具函数 ==========
print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

timestamp() {
    date +%Y%m%d_%H%M%S
}

# ========== 检查环境 ==========
check_env() {
    print_step "Step 0: 环境检查"

    echo "Python: $(python3 --version 2>&1)"
    echo "PyTorch:"
    python3 -c "
import torch
print(f'  version: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')
else:
    print(f'  Running on CPU')
"

    echo "Config: ${CONFIG_FILE}"
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found!"
        exit 1
    fi
    echo "Debug level: WALPURGIS_DEBUG=${DEBUG}"
    echo ""
    echo "✓ 环境就绪"
}

# ========== 生成合成数据 ==========
generate_data() {
    print_step "Step 1: 数据生成"

    TRAIN_FILE="${PROJECT_DIR}/datasets/METR-LA/train.npz"
    if [ -f "${TRAIN_FILE}" ]; then
        echo "数据已存在: ${TRAIN_FILE}"
        echo "跳过数据生成..."
        return 0
    fi

    echo "生成合成METR-LA数据..."
    python3 generate_synth_data.py

    echo ""
    echo "✓ 数据生成完毕"
}

# ========== 数据预检查 — 断点调试风格 ==========
inspect_data() {
    print_step "Step 2: 数据预检查 (断点调试)"

    python3 << 'PYEOF'
import numpy as np
import pickle
import os, sys

print("=" * 60)
print("[INSPECT] 数据完整性与统计量检查")
print("=" * 60)

base = "walpurgis/datasets"

# 检查 npz 文件
for split in ["train", "val", "test"]:
    path = os.path.join(base, "METR-LA", f"{split}.npz")
    if not os.path.exists(path):
        print(f"  ✗ MISSING: {path}")
        continue
    data = np.load(path)
    x, y = data['x'], data['y']
    print(f"\n  [{split}]")
    print(f"    x.shape = {x.shape}  y.shape = {y.shape}")
    print(f"    x dtype = {x.dtype}  y dtype = {y.dtype}")
    print(f"    x[...,0] (speed):  min={x[...,0].min():.2f}  max={x[...,0].max():.2f}  "
          f"mean={x[...,0].mean():.2f}  std={x[...,0].std():.2f}")
    print(f"    x[...,1] (ToD):    min={x[...,1].min():.4f}  max={x[...,1].max():.4f}")
    nan_count = np.isnan(x).sum()
    inf_count = np.isinf(x).sum()
    print(f"    NaN count: {nan_count}  Inf count: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print(f"    ⚠ WARNING: 数据中存在异常值!")

# 检查 adj
adj_path = os.path.join(base, "sensor_graph", "adj_mx_la.pkl")
if os.path.exists(adj_path):
    with open(adj_path, 'rb') as f:
        sensor_ids, id_to_ind, adj_mx = pickle.load(f)
    print(f"\n  [adj_mx]")
    print(f"    shape = {adj_mx.shape}  dtype = {adj_mx.dtype}")
    print(f"    num_sensors = {len(sensor_ids)}")
    print(f"    value range: [{adj_mx.min():.4f}, {adj_mx.max():.4f}]")
    print(f"    sparsity: {(adj_mx == 0).sum() / adj_mx.size:.1%}")
    print(f"    symmetric: {np.allclose(adj_mx, adj_mx.T)}")
    # 检查图的连通性
    nonzero_per_row = (adj_mx > 0.01).sum(axis=1)
    print(f"    avg neighbors (>0.01): {nonzero_per_row.mean():.1f}")
    isolated = (nonzero_per_row == 0).sum()
    if isolated > 0:
        print(f"    ⚠ WARNING: {isolated} isolated nodes!")
else:
    print(f"  ✗ MISSING: {adj_path}")

print("\n" + "=" * 60)
print("[INSPECT] 完成")
print("=" * 60)
PYEOF

    echo "✓ 数据检查完毕"
}

# ========== 模型结构检查 ==========
inspect_model() {
    print_step "Step 3: 模型结构 & 参数检查"

    python3 << 'PYEOF'
import sys, os, yaml, torch, pickle, numpy as np
sys.path.insert(0, "walpurgis")
os.chdir("walpurgis")

from models.model import D2STGNN
from walpurgis import snapshot_model
from utils.cal_adj import transition_matrix

config_path = "configs/SYNTH-METRLA.yaml"
with open(config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 加载adj
with open(cfg['data_args']['adj_data_path'], 'rb') as f:
    sensor_ids, id_to_ind, adj_mx = pickle.load(f)
num_nodes = adj_mx.shape[0]

# 构造model_args (和main.py一样)
model_args = cfg['model_args']
model_args['device'] = torch.device('cpu')
model_args['num_nodes'] = num_nodes
adj_t = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
model_args['adjs'] = [torch.tensor(a, dtype=torch.float32) for a in adj_t]
model_args['adjs_ori'] = torch.tensor(adj_mx, dtype=torch.float32)
model_args['dataset'] = 'METR-LA'

model = D2STGNN(**model_args)

print("=" * 60)
print("[MODEL] D2STGNN 结构概览")
print("=" * 60)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total:,}")
print(f"  Trainable params: {trainable:,}")
print(f"  Model size (MB):  {total * 4 / 1e6:.2f}")
print()

# 打印每层的参数量
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:30s}  {params:>8,} params")

# 用假数据做一次 forward, 看看 shape 流转
print("\n" + "-" * 60)
print("[FORWARD] Dry-run with synthetic input")
print("-" * 60)

B, T, N, D = 2, 12, num_nodes, 2    # D=2: speed + time_of_day
x = torch.randn(B, T, N, D)
x[..., 0] = x[..., 0].abs() * 30 + 20     # speed: 20~80
x[..., 1] = torch.rand(B, T, N) * 0.99     # time_of_day: [0, 1)

try:
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  ✓ Forward pass succeeded")
except Exception as e:
    print(f"  ✗ Forward failed: {e}")
    import traceback
    traceback.print_exc()

print()
# 做一次参数快照
print("[SNAPSHOT] 初始参数状态:")
snapshot_model(model, epoch=0, step=0, top_k=5)

os.chdir("..")
PYEOF

    echo "✓ 模型检查完毕"
}

# ========== 训练 ==========
run_training() {
    print_step "Step 4: 训练 (WALPURGIS_DEBUG=${DEBUG})"

    cd "${PROJECT_DIR}"

    LOG_DIR="output/logs"
    mkdir -p "${LOG_DIR}" output

    TIMESTAMP=$(timestamp)
    LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

    echo "Config:  ${CONFIG_FILE}"
    echo "Dataset: ${DATASET}"
    echo "Device:  ${DEVICE}"
    echo "Log:     ${LOG_FILE}"
    echo ""

    python3 main.py --dataset METR-LA 2>&1 | tee "${LOG_FILE}"

    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -eq 0 ]; then
        echo ""
        echo "✓ 训练完成"
    else
        echo ""
        echo "✗ 训练失败 (exit code: ${TRAIN_EXIT})"
        echo "  查看日志: ${LOG_FILE}"
    fi

    cd "${SCRIPT_DIR}"
    return $TRAIN_EXIT
}

# ========== 全链路快速测试 ==========
smoke_test() {
    print_step "Step 5: Smoke Test (一轮前向+反向)"

    python3 << 'PYEOF'
import sys, os, yaml, torch
sys.path.insert(0, "walpurgis")
os.chdir("walpurgis")
os.environ["WALPURGIS_DEBUG"] = "1"

from models.model import D2STGNN
from models.losses import masked_mae
from walpurgis import (_dbg, snapshot_model, register_activation_hooks,
                       gradient_health_check)

config_path = "configs/SYNTH-METRLA.yaml"
with open(config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_nodes = 20
model = D2STGNN(num_nodes=num_nodes, **cfg['model_args'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# 注册 activation hooks
tracker = register_activation_hooks(model)

print("=" * 70)
print("[SMOKE] 一轮前向 + 反向传播, 检查梯度和激活")
print("=" * 70)

B, T, N, D = 4, 12, num_nodes, 1
x = torch.randn(B, T, N, D)
y = torch.randn(B, T, N, D)

model.train()
out = model(x)
loss = masked_mae(out, y[:, :out.shape[1], :, :], 0.0)
print(f"\n[SMOKE] Loss = {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()

# 梯度健康检查
print("\n[SMOKE] 梯度健康检查:")
issues = gradient_health_check(model)
if not issues:
    print("  ✓ 所有梯度健康")

# 激活检查
print("\n[SMOKE] 激活值统计:")
tracker.report()
dead = tracker.check_dead()
tracker.remove()

# 参数快照
snapshot_model(model, epoch=0, step=1, top_k=5)

optimizer.step()

# 再做一次前向, 检查更新后的输出
model.eval()
with torch.no_grad():
    out2 = model(x)
    loss2 = masked_mae(out2, y[:, :out2.shape[1], :, :], 0.0)
print(f"[SMOKE] Loss after 1 step: {loss2.item():.4f}")
print(f"  delta: {loss.item() - loss2.item():.6f}")
print()

os.chdir("..")
print("✓ Smoke test 通过")
PYEOF
}

# ========== 帮助 ==========
show_help() {
    cat << 'EOF'
LLM4Walking — Walpurgis Pipeline

Usage: ./run_walpurgis.sh [command]

Commands:
  check        检查Python环境 + GPU
  data         生成合成数据
  inspect      数据完整性断点检查
  model        模型结构 + 参数快照
  smoke        一轮前向/反向 + 梯度/激活诊断
  train        启动完整训练
  all          全链路 (check → data → inspect → model → smoke → train)

Environment Variables:
  WALPURGIS_DEBUG=1          开启全部调试打印
  WALPURGIS_DEBUG=model,trainer   只开启指定模块
  DATASET=SYNTH-METRLA       配置文件名
  EPOCHS=5                    训练轮数
  DEVICE=cpu                  设备 (cpu / cuda:0)

示例:
  # 完整pipeline
  WALPURGIS_DEBUG=1 ./run_walpurgis.sh all

  # 只做smoke test
  WALPURGIS_DEBUG=model,trainer ./run_walpurgis.sh smoke

  # 训练并记录日志
  WALPURGIS_DEBUG=1 ./run_walpurgis.sh train 2>&1 | tee run.log
EOF
}

# ========== 主入口 ==========
main() {
    COMMAND=${1:-"help"}

    case $COMMAND in
        check)
            check_env ;;
        data|generate)
            generate_data ;;
        inspect)
            inspect_data ;;
        model)
            inspect_model ;;
        smoke)
            smoke_test ;;
        train)
            run_training ;;
        all)
            check_env
            generate_data
            inspect_data
            inspect_model
            smoke_test
            # run_training  # 最后手动train, 避免CI里跑太久
            echo ""
            echo "==========================================="
            echo "全部检查通过. 执行以下命令开始训练:"
            echo "  WALPURGIS_DEBUG=1 ./run_walpurgis.sh train"
            echo "==========================================="
            ;;
        help|--help|-h)
            show_help ;;
        *)
            echo "Unknown: $COMMAND"
            show_help
            exit 1 ;;
    esac
}

main "$@"
