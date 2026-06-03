"""PEMS08 training data generator — robust normalization + cyclic encoding.

Same algorithmic changes as PEMS04:
1. Robust MinMax (percentile-based bounds).
2. Cyclic (sin/cos) temporal encoding.
3. Variance-aware splitting.
"""

from __future__ import absolute_import, division, print_function

import argparse
import pickle
import numpy as np
import os
import math

num_feat = 1


def _cyclic_encode(values, period):
    angle = 2.0 * math.pi * values / period
    return np.sin(angle), np.cos(angle)


def RobustMinMaxNormalization(train, val, test, lo_pct=1, hi_pct=99):
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    _min = np.percentile(train, lo_pct, axis=(0, 1, 3), keepdims=True)
    _max = np.percentile(train, hi_pct, axis=(0, 1, 3), keepdims=True)
    spread = _max - _min
    spread[spread < 1e-6] = 1.0

    def normalize(x):
        x = np.clip(x, _min, _max)
        return 2.0 * ((x - _min) / spread) - 1.0

    return ({'_max': _max.astype(np.float32),
             '_min': _min.astype(np.float32)},
            normalize(train), normalize(val), normalize(test))


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]

    if add_time_in_day:
        time_frac = np.array([i % 288 / 288.0 for i in range(num_samples)])
        tid_sin, tid_cos = _cyclic_encode(time_frac, 1.0)
        tid_sin = np.tile(tid_sin, [1, num_nodes, 1]).transpose((2, 1, 0))
        tid_cos = np.tile(tid_cos, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(tid_sin)
        feature_list.append(tid_cos)

    if add_day_in_week:
        dow = np.array([(i // 288) % 7 for i in range(num_samples)],
                       dtype=np.float64)
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


def _variance_aware_split(x, y, train_r=0.6, val_r=0.2):
    n = x.shape[0]
    var = x[:, :, :, 0].var(axis=(1, 2))
    n_buckets = 10
    bucket_idx = np.clip(
        (np.argsort(np.argsort(var)) * n_buckets) // n,
        0, n_buckets - 1)

    train_m = np.zeros(n, dtype=bool)
    val_m = np.zeros(n, dtype=bool)
    test_m = np.zeros(n, dtype=bool)

    for b in range(n_buckets):
        idx = np.sort(np.where(bucket_idx == b)[0])
        nb = len(idx)
        nt = int(nb * train_r)
        nv = int(nb * val_r)
        train_m[idx[:nt]] = True
        val_m[idx[nt:nt + nv]] = True
        test_m[idx[nt + nv:]] = True

    return (x[train_m], y[train_m],
            x[val_m], y[val_m],
            x[test_m], y[test_m])


def generate_train_val_test(args):
    data = np.load(args.traffic_df_filename)['data']
    x_offsets = np.sort(np.concatenate(
        (np.arange(-(args.seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        data, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)

    print("x shape:", x.shape, ", y shape:", y.shape)

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = _variance_aware_split(x, y, 0.6, 0.2)

    y_train = y_train[..., 0:1]
    y_val = y_val[..., 0:1]
    y_test = y_test[..., 0:1]

    x_train_feat = np.transpose(x_train[:, :, :, :num_feat], [0, 2, 3, 1])
    x_val_feat = np.transpose(x_val[:, :, :, :num_feat], [0, 2, 3, 1])
    x_test_feat = np.transpose(x_test[:, :, :, :num_feat], [0, 2, 3, 1])

    stat, x_train_feat, x_val_feat, x_test_feat = \
        RobustMinMaxNormalization(x_train_feat, x_val_feat, x_test_feat)

    x_train_feat = np.transpose(x_train_feat, [0, 3, 1, 2])
    x_val_feat = np.transpose(x_val_feat, [0, 3, 1, 2])
    x_test_feat = np.transpose(x_test_feat, [0, 3, 1, 2])

    x_train = np.concatenate([x_train_feat, x_train[:, :, :, num_feat:]], -1)
    x_val = np.concatenate([x_val_feat, x_val[:, :, :, num_feat:]], -1)
    x_test = np.concatenate([x_test_feat, x_test[:, :, :, num_feat:]], -1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

    pickle.dump(stat['_max'], open("datasets/PEMS08/max.pkl", 'wb'))
    pickle.dump(stat['_min'], open("datasets/PEMS08/min.pkl", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='datasets/PEMS08')
    parser.add_argument("--traffic_df_filename", type=str,
                        default='datasets/raw_data/PEMS08/PEMS08.npz')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
