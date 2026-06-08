#!/usr/bin/env bash
# server_run_all.sh — 一键在GPU服务器上运行所有SOTA实验
# 在服务器上执行: bash server_run_all.sh
# 服务器: AMD EPYC 9354, 2xA6000+H100, conda base, CUDA 11.5
set -euo pipefail

REPO_DIR="/data/jiacheng/walpurgis-WTFGG"
CONDA_ENV="walpurgis"
GH_TOKEN="${GH_TOKEN:-}"  # 需要设置环境变量

echo "============================================"
echo "  Walpurgis SOTA Experiment Runner"
echo "  $(date)"
echo "============================================"

# 1. Clone或pull仓库
if [ -d "$REPO_DIR/.git" ]; then
    echo ">>> Pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo ">>> Cloning repo..."
    if [ -n "$GH_TOKEN" ]; then
        git clone "https://${GH_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" "$REPO_DIR"
    else
        git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git "$REPO_DIR"
    fi
    cd "$REPO_DIR"
fi

git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 2. Conda环境
eval "$(conda shell.bash hook)"
if conda env list | grep -qw "$CONDA_ENV"; then
    echo ">>> Activating existing env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
else
    echo ">>> Creating conda env: $CONDA_ENV"
    conda create -y -n "$CONDA_ENV" python=3.10
    conda activate "$CONDA_ENV"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scipy pyyaml scikit-learn tables pandas
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 3. 准备METR-LA数据集
DATA_DIR="$REPO_DIR/datasets"
mkdir -p "$DATA_DIR/METR-LA" "$DATA_DIR/PEMS-BAY" "$DATA_DIR/sensor_graph"

if [ ! -f "$DATA_DIR/METR-LA/train.npz" ]; then
    echo ">>> Preparing METR-LA dataset..."

    # 下载metr-la.h5
    if [ ! -f "$DATA_DIR/METR-LA/metr-la.h5" ]; then
        echo ">>> Downloading metr-la.h5..."
        pip install gdown 2>/dev/null
        # 尝试多个来源
        wget -q -O "$DATA_DIR/METR-LA/metr-la.h5" \
            "https://zenodo.org/records/5724362/files/metr-la.h5?download=1" 2>/dev/null || \
        gdown "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC" -O "$DATA_DIR/METR-LA/metr-la.h5" 2>/dev/null || \
        echo "ERROR: 无法下载metr-la.h5, 请手动放到 $DATA_DIR/METR-LA/metr-la.h5"
    fi

    # 下载adj_mx_la.pkl (sensor graph)
    if [ ! -f "$DATA_DIR/sensor_graph/adj_mx_la.pkl" ]; then
        echo ">>> Downloading adj_mx_la.pkl..."
        wget -q -O "$DATA_DIR/sensor_graph/adj_mx_la.pkl" \
            "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl" 2>/dev/null || \
        echo "WARNING: adj_mx_la.pkl下载失败, 请手动放置"
    fi

    # 生成训练数据
    python3 << 'GENEOF'
import numpy as np
import os
import pandas as pd

data_dir = os.environ.get('DATA_DIR', 'datasets') + '/METR-LA'
h5_path = os.path.join(data_dir, 'metr-la.h5')

if not os.path.exists(h5_path):
    print(f"ERROR: {h5_path} not found")
    exit(1)

df = pd.read_hdf(h5_path)
print(f"METR-LA: {df.shape} ({df.shape[0]} timesteps, {df.shape[1]} nodes)")

x_offsets = np.arange(-11, 1, 1)  # 12 input steps
y_offsets = np.arange(1, 13, 1)   # 12 output steps

num_samples = df.shape[0]
data = np.expand_dims(df.values, axis=-1)  # [T, N, 1]

# time_in_day
time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
time_in_day = np.tile(time_ind, [1, df.shape[1], 1]).transpose((2, 1, 0))
# day_in_week
dow = df.index.dayofweek / 7.0
dow_tiled = np.tile(dow, [1, df.shape[1], 1]).transpose((2, 1, 0))

data = np.concatenate([data, time_in_day, dow_tiled], axis=-1)  # [T, N, 3]
print(f"Features: {data.shape[-1]} (value, time_in_day, day_in_week)")

x_list, y_list = [], []
min_t = abs(min(x_offsets))
max_t = num_samples - abs(max(y_offsets))
for t in range(min_t, max_t):
    x_list.append(data[t + x_offsets, ...])
    y_list.append(data[t + y_offsets, ...])

x = np.stack(x_list, axis=0)
y = np.stack(y_list, axis=0)
print(f"Samples: x={x.shape}, y={y.shape}")

# 7:1:2 split
n = x.shape[0]
n_train = int(n * 0.7)
n_test = int(n * 0.2)
n_val = n - n_train - n_test

for name, sx, sy in [
    ('train', x[:n_train], y[:n_train]),
    ('val', x[n_train:n_train+n_val], y[n_train:n_train+n_val]),
    ('test', x[-n_test:], y[-n_test:]),
]:
    path = os.path.join(data_dir, f'{name}.npz')
    np.savez_compressed(path, x=sx, y=sy,
                        x_offsets=x_offsets.reshape(-1,1),
                        y_offsets=y_offsets.reshape(-1,1))
    print(f"  {name}: x={sx.shape} y={sy.shape} -> {path}")

print("METR-LA dataset ready.")
GENEOF
fi

# 4. 运行实验 — 按变体潜力排序
# H100 (cuda:2) 跑最重要的, A6000 (cuda:0/1) 跑其他的
VARIANTS=("cascade" "nebula" "prism" "flux" "reverie")
DATASETS=("METR-LA")
DEVICE="cuda:2"  # H100
EPOCHS=80

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Running: walpurgis_${variant} on ${DATASETS[0]}"
    echo "  Device: $DEVICE | Epochs: $EPOCHS"
    echo "  $(date)"
    echo "============================================"

    TRAIN_SCRIPT="train_${variant}.py"
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "SKIP: $TRAIN_SCRIPT not found"
        continue
    fi

    RESULT_DIR="output/results_${variant}_${DATASETS[0]}"
    mkdir -p "$RESULT_DIR"

    # 运行训练 (单seed先跑通)
    python "$TRAIN_SCRIPT" \
        --dataset "${DATASETS[0]}" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        2>&1 | tee "${RESULT_DIR}/train.log"

    # 提取结果
    python3 -c "
import re
with open('${RESULT_DIR}/train.log') as f:
    lines = f.readlines()
for line in reversed(lines):
    m = re.search(r'Avg MAE:\s*([\d.]+).*Avg RMSE:\s*([\d.]+).*Avg MAPE:\s*([\d.]+)', line)
    if m:
        print(f'RESULT: ${variant} on ${DATASETS[0]}: MAE={m.group(1)} RMSE={m.group(2)} MAPE={m.group(3)}%')
        break
"

    # Push结果
    if [ -n "$GH_TOKEN" ]; then
        git add "${RESULT_DIR}/" output/*.pt output/*.json 2>/dev/null || true
        git commit -m "exp: walpurgis_${variant} on ${DATASETS[0]} — $(date +%Y%m%d_%H%M)" || true
        git push "https://${GH_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git" main || true
    fi

    echo ">>> ${variant} complete: $(date)"
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  $(date)"
echo "============================================"

# 5. 生成汇总
python3 << 'SUMEOF'
import os, json, re, glob

results = {}
for log_path in glob.glob("output/results_*/train.log"):
    variant = log_path.split("/")[1].replace("results_", "").rsplit("_", 1)[0]
    dataset = log_path.split("/")[1].rsplit("_", 1)[-1]
    with open(log_path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        m = re.search(r'Avg MAE:\s*([\d.]+).*Avg RMSE:\s*([\d.]+).*Avg MAPE:\s*([\d.]+)', line)
        if m:
            results[f"{variant}_{dataset}"] = {
                "variant": variant, "dataset": dataset,
                "MAE": float(m.group(1)),
                "RMSE": float(m.group(2)),
                "MAPE": float(m.group(3))
            }
            break

# SOTA reference
sota = {"STAEFormer": {"MAE": 2.90, "RMSE": 5.91, "MAPE": 8.12},
        "D2STGNN": {"MAE": 3.04, "RMSE": 6.23, "MAPE": 8.33}}

print("\n" + "="*60)
print("  RESULTS SUMMARY vs SOTA")
print("="*60)
for k, v in sorted(results.items()):
    beats = "BEATS SOTA!" if v["MAE"] < 2.90 else ""
    print(f"  {v['variant']:15s} {v['dataset']:10s} MAE={v['MAE']:.4f} RMSE={v['RMSE']:.4f} MAPE={v['MAPE']:.2f}% {beats}")
print("-"*60)
print(f"  {'STAEFormer':15s} {'METR-LA':10s} MAE=2.90   RMSE=5.91   MAPE=8.12%  (current SOTA)")
print("="*60)

with open("output/all_results_summary.json", "w") as f:
    json.dump({"results": results, "sota": sota}, f, indent=2)
print("Summary saved to output/all_results_summary.json")
SUMEOF
