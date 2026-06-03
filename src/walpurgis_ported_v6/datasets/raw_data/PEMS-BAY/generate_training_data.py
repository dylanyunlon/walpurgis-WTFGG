"""PEMS-BAY training data generator — cyclic temporal encoding + variance-aware split.

Same algorithmic changes as METR-LA version:
1. Cyclic (sin/cos) time encoding replaces linear fraction.
2. Variance-aware splitting replaces strict temporal cut.
"""

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os
import pandas as pd
import math


def _cyclic_encode(values, period):
    angle = 2.0 * math.pi * values / period
    return np.sin(angle), np.cos(angle)


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        time_frac = (
            (df.index.values - df.index.values.astype("datetime64[D]"))
            / np.timedelta64(1, "D")
        )
        tid_sin, tid_cos = _cyclic_encode(time_frac, 1.0)
        tid_sin = np.tile(tid_sin, [1, num_nodes, 1]).transpose((2, 1, 0))
        tid_cos = np.tile(tid_cos, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(tid_sin)
        feature_list.append(tid_cos)

    if add_day_in_week:
        dow = df.index.dayofweek.values.astype(np.float64)
        dow_sin, dow_cos = _cyclic_encode(dow, 7.0)
        dow_sin = np.tile(dow_sin, [1, num_nodes, 1]).transpose((2, 1, 0))
        dow_cos = np.tile(dow_cos, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_sin)
        feature_list.append(dow_cos)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return np.stack(x, axis=0), np.stack(y, axis=0)


def _variance_aware_split(x, y, train_r=0.7, val_r=0.1):
    n = x.shape[0]
    var = x[:, :, :, 0].var(axis=(1, 2))
    n_buckets = 10
    bucket_idx = np.clip(
        (np.argsort(np.argsort(var)) * n_buckets) // n,
        0, n_buckets - 1)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for b in range(n_buckets):
        idx = np.sort(np.where(bucket_idx == b)[0])
        nb = len(idx)
        n_train = int(nb * train_r)
        n_val = int(nb * val_r)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return (x[train_mask], y[train_mask],
            x[val_mask], y[val_mask],
            x[test_mask], y[test_mask])


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    x_offsets = np.sort(np.concatenate(
        (np.arange(-(args.seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)

    print("x shape:", x.shape, ", y shape:", y.shape)

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = _variance_aware_split(x, y, 0.7, 0.1)

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
    parser.add_argument("--output_dir", type=str, default='datasets/PEMS-BAY')
    parser.add_argument("--traffic_df_filename", type=str,
                        default='datasets/raw_data/PEMS-BAY/pems-bay.h5')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
