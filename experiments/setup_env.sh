#!/usr/bin/env bash
# experiments/setup_env.sh — 在 GPU 服务器上配置 conda 环境
# 用法: bash experiments/setup_env.sh
# 服务器: 2x A6000 (48GB) + 1x H100 NVL (96GB), EPYC 9354, CUDA 11.5+
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-walpurgis}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo " Walpurgis Environment Setup"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Repo: $REPO_DIR"
echo "============================================"

# ── 1. 系统信息 ────────────────────────────────────────
echo "[1/5] System info..."
lscpu | grep -E "Model name|CPU\(s\):" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "CUDA toolkit: $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"

# ── 2. Conda 环境 ──────────────────────────────────────
echo "[2/5] Conda environment '$CONDA_ENV'..."
if conda info --envs 2>/dev/null | grep -qw "$CONDA_ENV"; then
    echo "  Reusing existing environment"
else
    echo "  Creating new environment..."
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

set +u  # conda activate 内部引用未定义变量 PS1
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

# ── 3. 依赖安装 ────────────────────────────────────────
echo "[3/5] Installing dependencies..."
# 检测最高可用 CUDA 版本 (driver 550 支持 CUDA 12.4)
CUDA_TAG="cu121"
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2\."; then
    echo "  PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    pip install torch==2.3.1 torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi
pip install -q numpy scipy pandas pyyaml setproctitle matplotlib tensorboard scikit-learn h5py tables 2>&1 | tail -3

# ── 4. 下载数据 ────────────────────────────────────────
echo "[4/5] Preparing datasets..."
cd "$REPO_DIR"

# METR-LA
if [ -f "datasets/METR-LA/train.npz" ]; then
    echo "  METR-LA: already prepared"
else
    echo "  METR-LA: running prepare_metrla.sh..."
    bash prepare_metrla.sh
fi

# PEMS-BAY (同结构, 从 DCRNN 官方源下载)
if [ -f "datasets/PEMS-BAY/train.npz" ]; then
    echo "  PEMS-BAY: already prepared"
else
    echo "  PEMS-BAY: download needed (see README)"
fi

# ── 5. 验证 ────────────────────────────────────────────
echo "[5/5] Verification..."
python3 -c "
import torch, numpy, scipy, yaml
print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}  Available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU{i}: {p.name}  {p.total_mem/(1024**3):.0f}GB  cc={p.major}.{p.minor}')
import numpy as np
for ds in ['METR-LA', 'PEMS-BAY']:
    try:
        d = np.load(f'datasets/{ds}/test.npz')
        print(f'{ds}: x={d[\"x\"].shape}  y={d[\"y\"].shape}')
    except:
        print(f'{ds}: NOT READY')
"

echo ""
echo "============================================"
echo " Setup complete. Activate with:"
echo "   conda activate $CONDA_ENV"
echo "============================================"
