"""
generate_training_data.py (METR-LA) — v9 port
Algo delta:
  1. 时间特征: upstream 用线性 fraction ∈ [0,1)
     → v9 改 cyclic sin/cos 双通道:
       sin(2π·t/288), cos(2π·t/288)
     保留 23:55→00:00 的周期连续性 (线性在此处有断崖)
  2. 数据划分: upstream 固定 70/10/20 按时间序顺切
     → v9 按每个样本的方差排序后交替抽样(stratified by variance),
     确保 train/val/test 三个 split 的方差分布一致,
     减少因为时段差异导致的分布漂移
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os, math
import numpy as np
import pandas as pd


def _cyclic_time(frac, period):
    """sin/cos cyclic encoding."""
    angle = 2.0 * math.pi * frac / period
    return math.sin(angle), math.cos(angle)


def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        # v9: cyclic sin/cos time-of-day (2 channels)
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        sin_tod = np.sin(2.0 * np.pi * time_ind)
        cos_tod = np.cos(2.0 * np.pi * time_ind)
        feature_list.append(np.tile(sin_tod, [1, num_nodes, 1]).transpose((2, 1, 0)))
        feature_list.append(np.tile(cos_tod, [1, num_nodes, 1]).transpose((2, 1, 0)))

    if add_day_in_week:
        # v9: cyclic sin/cos day-of-week (2 channels)
        dow = df.index.dayofweek.values.astype(float)
        sin_dow = np.sin(2.0 * np.pi * dow / 7.0)
        cos_dow = np.cos(2.0 * np.pi * dow / 7.0)
        feature_list.append(np.tile(sin_dow, [1, num_nodes, 1]).transpose((2, 1, 0)))
        feature_list.append(np.tile(cos_dow, [1, num_nodes, 1]).transpose((2, 1, 0)))

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return np.stack(x, axis=0), np.stack(y, axis=0)


def _variance_stratified_split(x, y, ratios=(0.7, 0.1, 0.2)):
    """
    v9: 按样本方差排序后交替抽样到三个 split,
    保证每个 split 内方差分布近似一致.
    """
    n = x.shape[0]
    # 每个样本的方差 (只看特征通道0)
    var_per_sample = np.var(x[:, :, :, 0].reshape(n, -1), axis=1)
    sorted_idx = np.argsort(var_per_sample)

    # 交替分配: 0→train, 1→val, 2→test, 然后循环
    n_train = round(n * ratios[0])
    n_val   = round(n * ratios[1])
    # 按排好序的索引, 每3个一组分配
    train_idx, val_idx, test_idx = [], [], []
    buckets = [train_idx, val_idx, test_idx]
    targets = [n_train, n_val, n - n_train - n_val]
    bi = 0
    for i, idx in enumerate(sorted_idx):
        # round-robin, 但尊重每个 bucket 的配额
        placed = False
        for attempt in range(3):
            b = (bi + attempt) % 3
            if len(buckets[b]) < targets[b]:
                buckets[b].append(idx)
                bi = (b + 1) % 3
                placed = True
                break
        if not placed:
            # 放到还没满的 bucket
            for b in range(3):
                if len(buckets[b]) < targets[b]:
                    buckets[b].append(idx)
                    break

    # 各 split 内部按时间顺序排列
    train_idx = sorted(train_idx)
    val_idx   = sorted(val_idx)
    test_idx  = sorted(test_idx)

    print(f"v9 stratified split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"  var(train)={np.mean(var_per_sample[train_idx]):.4f}  "
          f"var(val)={np.mean(var_per_sample[val_idx]):.4f}  "
          f"var(test)={np.mean(var_per_sample[test_idx]):.4f}")

    return (x[train_idx], y[train_idx],
            x[val_idx],   y[val_idx],
            x[test_idx],  y[test_idx])


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)

    print("x shape:", x.shape, ", y shape:", y.shape)

    # v9: variance-stratified split
    x_train, y_train, x_val, y_val, x_test, y_test = _variance_stratified_split(x, y)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/METR-LA")
    parser.add_argument("--traffic_df_filename", type=str, default="datasets/raw_data/METR-LA/metr-la.h5")
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
