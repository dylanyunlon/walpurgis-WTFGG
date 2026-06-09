#!/usr/bin/env python3
"""
prepare_metrla.py — 从原始 metr-la.h5 生成训练用 npz + 邻接矩阵
依赖: numpy, pandas, h5py, pickle (标准库)
用法: python prepare_metrla.py [--raw_dir DIR] [--out_dir DIR]

生成:
  datasets/METR-LA/{train,val,test}.npz   — x:(N,12,207,3) y:(N,12,207,1)
  datasets/sensor_graph/adj_mx_la.pkl     — (ids, id2idx, adj_matrix)
"""
import argparse
import os
import pickle
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

# ── 依赖检测 ──────────────────────────────────────────────
def check_deps():
    missing = []
    for mod in ['h5py', 'numpy', 'pandas']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Fix:   pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()
import h5py


# ── 路径常量 ──────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW  = os.path.join(REPO, 'src', 'walpurgis', 'datasets', 'raw_data', 'METR-LA')
DEFAULT_ADJ  = os.path.join(REPO, 'datasets', 'sensor_graph')
DEFAULT_OUT  = os.path.join(REPO, 'datasets', 'METR-LA')
DOWNLOAD_URL = 'https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download'

SEQ_X = 12  # 输入窗口长度
SEQ_Y = 12  # 预测窗口长度
TRAIN_RATIO = 0.7
TEST_RATIO  = 0.2


def download(raw_dir, adj_dir):
    """下载并解压原始 METR-LA 数据"""
    h5_path = os.path.join(raw_dir, 'metr-la.h5')
    ids_path = os.path.join(adj_dir, 'sensor_ids_la.txt')
    dist_path = os.path.join(adj_dir, 'distances_la.csv')

    # 检查旧路径 (之前的 bash 脚本写到 src/walpurgis/datasets/sensor_graph/)
    old_adj_dir = os.path.join(os.path.dirname(raw_dir), 'sensor_graph')
    for fname in ['sensor_ids_la.txt', 'distances_la.csv', 'sensor_locations_la.csv']:
        dst = os.path.join(adj_dir, fname)
        old = os.path.join(old_adj_dir, fname)
        if not os.path.isfile(dst) and os.path.isfile(old):
            import shutil
            shutil.copy2(old, dst)
            print(f'  Migrated {fname} from old path')

    # 全部文件就位则跳过
    if (os.path.isfile(h5_path) and os.path.getsize(h5_path) > 1_000_000
            and os.path.isfile(ids_path) and os.path.isfile(dist_path)):
        print('  All source files present, skipping download')
        return h5_path

    zip_path = '/tmp/metrla_download.zip'
    print(f'  Downloading from {DOWNLOAD_URL} ...')
    subprocess.check_call(['wget', '-q', DOWNLOAD_URL, '-O', zip_path])

    print('  Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('/tmp/metrla_extract')

    # 复制文件到目标位置
    import shutil
    for name, dst_dir in [
        ('metr_la.h5', raw_dir),
        ('distances_la.csv', adj_dir),
        ('sensor_ids_la.txt', adj_dir),
        ('sensor_locations_la.csv', adj_dir),
    ]:
        src = os.path.join('/tmp/metrla_extract', name)
        if not os.path.exists(src):
            # 有些 zip 没有子目录
            for root, _, files in os.walk('/tmp/metrla_extract'):
                if name in files:
                    src = os.path.join(root, name)
                    break
        if os.path.exists(src):
            dst = os.path.join(dst_dir, name.replace('metr_la', 'metr-la'))
            shutil.copy2(src, dst)

    # 清理
    os.remove(zip_path)
    shutil.rmtree('/tmp/metrla_extract', ignore_errors=True)
    return os.path.join(raw_dir, 'metr-la.h5')


def read_h5(h5_path, adj_dir):
    """读取 metr-la.h5 中的交通速度数据。

    pandas HDFStore 的 axis0/axis1 使用 PyTables 特有编码,
    h5py 无法解码 (TypeError: Unsupported integer size (0))。
    只读 block0_values (纯浮点矩阵), 其余元数据从外部文件获取:
      - sensor_ids   <- sensor_ids_la.txt
      - timestamps   <- 已知起始日期 + 5min 间隔推算

    返回: data (T, N), sensor_ids (list of str), timestamps (datetime64)
    """
    # 1. 速度数据: block0_values 是标准 float64 数组, h5py 可读
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f'Empty HDF5 file: {h5_path}')
        grp = f[keys[0]]
        data = grp['block0_values'][:].astype(np.float32)  # (T, N)

    # 2. sensor IDs: 从下载时已解压的 sensor_ids_la.txt 读取
    ids_path = os.path.join(adj_dir, 'sensor_ids_la.txt')
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(
            f'Missing {ids_path} — rerun with download step')
    with open(ids_path) as f:
        content = f.read().strip()
    # 格式: 逗号分隔的 sensor ID 在一行里
    sensor_ids = [s.strip() for s in content.split(',') if s.strip()]

    if len(sensor_ids) != data.shape[1]:
        raise ValueError(
            f'sensor_ids count ({len(sensor_ids)}) != data columns ({data.shape[1]})')

    # 3. timestamps: METR-LA 从 2012-03-01 00:00 开始, 每 5 分钟一条
    start = np.datetime64('2012-03-01T00:00')
    timestamps = start + np.arange(data.shape[0]) * np.timedelta64(5, 'm')

    print(f'  Loaded: {data.shape[0]} timesteps, {data.shape[1]} sensors')
    print(f'  Range:  {timestamps[0]} ~ {timestamps[-1]}')
    return data, sensor_ids, timestamps


def build_adj(sensor_ids, adj_dir):
    """从 distances_la.csv 构建邻接矩阵"""
    csv_path = os.path.join(adj_dir, 'distances_la.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Missing: {csv_path}')

    id2idx = {sid: i for i, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)
    dist_mx = np.zeros((n, n), dtype=np.float64)

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        fi = str(int(row['from']))
        ti = str(int(row['to']))
        if fi in id2idx and ti in id2idx:
            dist_mx[id2idx[fi], id2idx[ti]] = float(row['cost'])

    # Gaussian kernel
    nonzero = dist_mx[dist_mx > 0]
    sigma = nonzero.std()
    adj = np.exp(-dist_mx ** 2 / sigma ** 2)
    adj[dist_mx == 0] = 0
    np.fill_diagonal(adj, 0)
    adj[adj < 0.1] = 0

    pkl_path = os.path.join(adj_dir, 'adj_mx_la.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump((sensor_ids, id2idx, adj), f, protocol=2)

    n_edges = int((adj > 0).sum())
    print(f'  Adjacency: {adj.shape}, {n_edges} edges -> {pkl_path}')
    return adj


def build_npz(data, timestamps, out_dir):
    """滑窗切片 + 特征拼接 (speed, time-of-day, day-of-week)"""
    T, N = data.shape

    # time-of-day: 用 timestamp 精确计算
    tod = (timestamps - timestamps.astype('datetime64[D]')) / np.timedelta64(1, 'D')
    tod = tod.astype(np.float32)

    # day-of-week: 0=Monday ... 6=Sunday
    # numpy datetime64 的 weekday: Monday=0
    # 用 pandas 更可靠
    dow = pd.DatetimeIndex(timestamps).dayofweek.values.astype(np.float32)

    # (T, N, 3): [speed, tod, dow]
    features = np.stack([
        data,
        np.tile(tod.reshape(-1, 1), (1, N)),
        np.tile(dow.reshape(-1, 1), (1, N)),
    ], axis=-1)  # (T, N, 3)

    # 滑窗
    x_offsets = np.arange(-(SEQ_X - 1), 1)      # [-11, -10, ..., 0]
    y_offsets = np.arange(1, SEQ_Y + 1)          # [1, 2, ..., 12]
    valid_idx = np.arange(SEQ_X - 1, T - SEQ_Y)

    x = features[valid_idx[:, None] + x_offsets[None, :]]  # (N_samples, 12, 207, 3)
    y = features[valid_idx[:, None] + y_offsets[None, :]]   # (N_samples, 12, 207, 3)

    # train/val/test split
    n_samples = len(x)
    n_train = round(n_samples * TRAIN_RATIO)
    n_test  = round(n_samples * TEST_RATIO)
    n_val   = n_samples - n_train - n_test

    splits = {
        'train': (x[:n_train],                y[:n_train]),
        'val':   (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test':  (x[-n_test:],                y[-n_test:]),
    }

    for name, (xd, yd) in splits.items():
        path = os.path.join(out_dir, f'{name}.npz')
        np.savez_compressed(path, x=xd, y=yd)
        print(f'  [{name}] x={xd.shape}  y={yd.shape}  -> {path}')


def main():
    parser = argparse.ArgumentParser(description='Prepare METR-LA dataset')
    parser.add_argument('--raw_dir', default=DEFAULT_RAW)
    parser.add_argument('--adj_dir', default=DEFAULT_ADJ)
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--skip_download', action='store_true')
    args = parser.parse_args()

    for d in [args.raw_dir, args.adj_dir, args.out_dir]:
        os.makedirs(d, exist_ok=True)

    # Step 1: Download
    print('[1/3] Download...')
    if args.skip_download:
        h5_path = os.path.join(args.raw_dir, 'metr-la.h5')
    else:
        h5_path = download(args.raw_dir, args.adj_dir)

    # Step 2: Adjacency matrix
    print('[2/3] Adjacency matrix...')
    data, sensor_ids, timestamps = read_h5(h5_path, args.adj_dir)
    build_adj(sensor_ids, args.adj_dir)

    # Step 3: Train/val/test NPZ
    print('[3/3] Sliding window NPZ...')
    build_npz(data, timestamps, args.out_dir)

    print('Done.')


if __name__ == '__main__':
    main()
