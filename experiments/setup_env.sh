#!/usr/bin/env bash
# experiments/setup_env.sh — 在 GPU 服务器上配置 conda 环境
# 用法: bash experiments/setup_env.sh
# 服务器: 2x A6000 (48GB) + 1x H100 NVL (96GB), EPYC 9354, CUDA 11.5+
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-walking3}"
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

# ── 2. 激活已有 conda 环境 ─────────────────────────────
echo "[2/4] Activating conda '$CONDA_ENV'..."
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'missing')"

# ── 3. 下载数据 ────────────────────────────────────────
echo "[3/4] Preparing datasets..."
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

# ── 4. 验证 ────────────────────────────────────────────
echo "[4/4] Verification..."
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
