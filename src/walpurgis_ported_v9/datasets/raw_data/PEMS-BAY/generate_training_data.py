"""
generate_training_data.py (PEMS-BAY) — v9 port
Algo delta (same as METR-LA):
  1. cyclic sin/cos 时间编码 (替代线性 fraction)
  2. variance-aware stratified split (替代固定 70/10/20)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os, math
import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        feature_list.append(np.tile(np.sin(2 * np.pi * time_ind), [1, num_nodes, 1]).transpose((2, 1, 0)))
        feature_list.append(np.tile(np.cos(2 * np.pi * time_ind), [1, num_nodes, 1]).transpose((2, 1, 0)))

    if add_day_in_week:
        dow = df.index.dayofweek.values.astype(float)
        feature_list.append(np.tile(np.sin(2 * np.pi * dow / 7), [1, num_nodes, 1]).transpose((2, 1, 0)))
        feature_list.append(np.tile(np.cos(2 * np.pi * dow / 7), [1, num_nodes, 1]).transpose((2, 1, 0)))

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return np.stack(x, axis=0), np.stack(y, axis=0)


def _variance_stratified_split(x, y, ratios=(0.7, 0.1, 0.2)):
    n = x.shape[0]
    var_per_sample = np.var(x[:, :, :, 0].reshape(n, -1), axis=1)
    sorted_idx = np.argsort(var_per_sample)
    n_train = round(n * ratios[0])
    n_val   = round(n * ratios[1])
    train_idx, val_idx, test_idx = [], [], []
    buckets = [train_idx, val_idx, test_idx]
    targets = [n_train, n_val, n - n_train - n_val]
    bi = 0
    for idx in sorted_idx:
        for attempt in range(3):
            b = (bi + attempt) % 3
            if len(buckets[b]) < targets[b]:
                buckets[b].append(idx)
                bi = (b + 1) % 3
                break
    return (x[sorted(train_idx)], y[sorted(train_idx)],
            x[sorted(val_idx)],   y[sorted(val_idx)],
            x[sorted(test_idx)],  y[sorted(test_idx)])


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    x_offsets = np.sort(np.concatenate((np.arange(-(args.seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)
    print("x shape:", x.shape, ", y shape:", y.shape)
    x_train, y_train, x_val, y_val, x_test, y_test = _variance_stratified_split(x, y)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(args.output_dir, f"{cat}.npz"),
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/PEMS-BAY")
    parser.add_argument("--traffic_df_filename", type=str, default="datasets/raw_data/PEMS-BAY/pems-bay.h5")
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
