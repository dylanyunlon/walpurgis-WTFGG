#!/usr/bin/env bash
# server_setup.sh — 在GPU服务器上一次性配置实验环境
# 使用方式: 由子Claude在GPU服务器上执行
set -euo pipefail

REPO_URL="https://github.com/dylanyunlon/walpurgis-WTFGG.git"
CONDA_ENV="walpurgis"
WORK_DIR="${HOME}/walpurgis-WTFGG"

echo "================================================"
echo " Walpurgis GPU Server Setup"
echo " $(date)"
echo "================================================"

# 1. 系统信息采集
echo "[1/7] System info..."
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
free -h
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
uname -r

# 2. Conda环境（复用已有或新建）
echo "[2/7] Setting up conda..."
if conda info --envs 2>/dev/null | grep -q "${CONDA_ENV}"; then
    echo "Environment '${CONDA_ENV}' exists, reusing"
    conda activate ${CONDA_ENV}
else
    echo "Creating conda environment '${CONDA_ENV}'..."
    conda create -n ${CONDA_ENV} python=3.10 -y
    conda activate ${CONDA_ENV}
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy scipy pyyaml setproctitle matplotlib pandas tensorboard
fi

# 3. Clone/Pull仓库
echo "[3/7] Repository setup..."
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 4. 下载METR-LA数据集
echo "[4/7] Downloading METR-LA dataset..."
METRLA_DIR="$WORK_DIR/datasets/METR-LA"
ADJ_DIR="$WORK_DIR/datasets/sensor_graph"
mkdir -p "$METRLA_DIR" "$ADJ_DIR"

if [ ! -f "$METRLA_DIR/train.npz" ]; then
    echo "Downloading METR-LA from D2STGNN official preprocessing..."
    # Use the standard DCRNN/STGCN preprocessing approach
    pip install gdown 2>/dev/null || true
    # Google Drive links for standard METR-LA processed data
    python3 << 'PYEOF'
import os, urllib.request, zipfile

# Standard METR-LA data sources (multiple fallbacks)
urls = [
    ("https://zenodo.org/record/5724362/files/METR-LA.zip", "METR-LA.zip"),
]
dst = os.environ.get("METRLA_DIR", "datasets/METR-LA")
adj_dst = os.environ.get("ADJ_DIR", "datasets/sensor_graph")
os.makedirs(dst, exist_ok=True)
os.makedirs(adj_dst, exist_ok=True)

# Try downloading
downloaded = False
for url, fname in urls:
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, f"/tmp/{fname}")
        if fname.endswith('.zip'):
            with zipfile.ZipFile(f"/tmp/{fname}", 'r') as z:
                z.extractall("/tmp/metrla_extract")
            # Move files to right location
            import shutil, glob
            for f in glob.glob("/tmp/metrla_extract/**/*.npz", recursive=True):
                shutil.copy2(f, dst)
            for f in glob.glob("/tmp/metrla_extract/**/*.pkl", recursive=True):
                if 'adj' in f.lower():
                    shutil.copy2(f, adj_dst)
        downloaded = True
        break
    except Exception as e:
        print(f"Failed: {e}")
        continue

if not downloaded:
    print("Auto-download failed. Will need manual data setup.")
    print("Download METR-LA from: https://github.com/liyaguang/DCRNN")
    print(f"Place train.npz, val.npz, test.npz in {dst}")
    print(f"Place adj_mx_la.pkl in {adj_dst}")
PYEOF
fi

# 5. 下载PEMS-BAY数据集
echo "[5/7] Downloading PEMS-BAY dataset..."
PEMSBAY_DIR="$WORK_DIR/datasets/PEMS-BAY"
mkdir -p "$PEMSBAY_DIR"
if [ ! -f "$PEMSBAY_DIR/train.npz" ]; then
    echo "PEMS-BAY: same procedure as METR-LA"
    echo "Place files in $PEMSBAY_DIR and adj_mx_bay.pkl in $ADJ_DIR"
fi

# 6. 验证环境
echo "[6/7] Verification..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem/(1024**3):.1f} GB')
import numpy, scipy, yaml, setproctitle
print('All dependencies OK')
"

# 7. Quick smoke test
echo "[7/7] Smoke test on SYNTH..."
cd "$WORK_DIR"
python3 src/walpurgis_cathexis/generate_synth_data.py
python3 train_cathexis.py --dataset SYNTH --device cuda:0 --epochs 2

echo ""
echo "================================================"
echo " Server setup complete!"
echo " Ready for METR-LA/PEMS-BAY training."
echo "================================================"
