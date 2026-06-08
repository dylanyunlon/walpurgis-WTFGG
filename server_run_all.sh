#!/usr/bin/env bash
# server_run_all.sh — 在GPU服务器上运行唯一变体walpurgis的METR-LA实验
# Usage: bash server_run_all.sh
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="walpurgis"
GH_TOKEN="${GH_TOKEN:-}"

echo "============================================"
echo "  Walpurgis SOTA Experiment Runner"
echo "  $(date)"
echo "============================================"

cd "$REPO_DIR"
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 2. Conda
export PS1="${PS1:-}"
eval "$(conda shell.bash hook)" 2>/dev/null || true
if conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
else
    conda create -y -n "$CONDA_ENV" python=3.10
    conda activate "$CONDA_ENV"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scipy pyyaml scikit-learn tables pandas
fi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 3. METR-LA数据集
DATA_DIR="$REPO_DIR/datasets"
mkdir -p "$DATA_DIR/METR-LA" "$DATA_DIR/sensor_graph"

if [ ! -f "$DATA_DIR/METR-LA/train.npz" ]; then
    echo ">>> Preparing METR-LA..."
    if [ ! -f "$DATA_DIR/METR-LA/metr-la.h5" ]; then
        pip install gdown 2>/dev/null
        gdown "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC" -O "$DATA_DIR/METR-LA/metr-la.h5" 2>/dev/null || \
        wget -q -O "$DATA_DIR/METR-LA/metr-la.h5" "https://zenodo.org/records/5724362/files/metr-la.h5?download=1"
    fi
    if [ ! -f "$DATA_DIR/sensor_graph/adj_mx_la.pkl" ]; then
        wget -q -O "$DATA_DIR/sensor_graph/adj_mx_la.pkl" \
            "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl"
    fi
    python3 -c "
import numpy as np, pandas as pd, os
data_dir = '$DATA_DIR/METR-LA'
df = pd.read_hdf(os.path.join(data_dir, 'metr-la.h5'))
print(f'METR-LA: {df.shape}')
data = np.expand_dims(df.values, axis=-1)
tid = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1,'D')
tid = np.tile(tid, [1, df.shape[1], 1]).transpose((2,1,0))
dow = df.index.dayofweek / 7.0
dow = np.tile(dow, [1, df.shape[1], 1]).transpose((2,1,0))
data = np.concatenate([data, tid, dow], axis=-1)
x_off = np.arange(-11, 1, 1); y_off = np.arange(1, 13, 1)
xs, ys = [], []
for t in range(11, len(df)-12):
    xs.append(data[t + x_off]); ys.append(data[t + y_off])
x, y = np.stack(xs), np.stack(ys)
n = x.shape[0]; nt = int(n*0.7); nv = int(n*0.1)
for nm, sx, sy in [('train',x[:nt],y[:nt]),('val',x[nt:nt+nv],y[nt:nt+nv]),('test',x[nt+nv:],y[nt+nv:])]:
    np.savez_compressed(os.path.join(data_dir, f'{nm}.npz'), x=sx, y=sy)
    print(f'  {nm}: {sx.shape}')
"
fi

# 4. 运行唯一变体 walpurgis
DEVICE="cuda:2"  # H100
EPOCHS=80

echo ""
echo "============================================"
echo "  Running: walpurgis on METR-LA"
echo "  Device: $DEVICE | Epochs: $EPOCHS"
echo "============================================"

python train_walpurgis.py --dataset METR-LA --device "$DEVICE" --epochs "$EPOCHS" 2>&1 | tee output/train_walpurgis_metrla.log

# 5. Push结果
if [ -n "$GH_TOKEN" ]; then
    git add output/ log/ 2>/dev/null || true
    git commit -m "exp: walpurgis on METR-LA — $(date +%Y%m%d_%H%M)" || true
    git push "https://${GH_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" main || true
fi

echo "Done. $(date)"
