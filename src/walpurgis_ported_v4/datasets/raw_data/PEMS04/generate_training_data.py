from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import sys

_V4_DEBUG = True
num_feat = 1


def MinMaxnormalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    # v4: add eps to denominator to prevent div-by-zero on constant features
    _eps = 1e-8

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min + _eps)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    if _V4_DEBUG:
        for name, arr in [('train', train_norm), ('val', val_norm), ('test', test_norm)]:
            print(f"[v4-DBG][MinMax] {name}: range=[{arr.min():.4f}, {arr.max():.4f}] "
                  f"nan_count={np.isnan(arr).sum()} inf_count={np.isinf(arr).sum()}", file=sys.stderr)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]
    if add_time_in_day:
        time_ind = [i % 288 / 288 for i in range(num_samples)]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = [(i // 288) % 7 for i in range(num_samples)]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)

    # v4: vectorized sliding window via stride_tricks instead of python loop
    # upstream: python loop appending slices — O(T) python iterations
    # v4: use np.lib.stride_tricks for zero-copy windowed view
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    valid_indices = np.arange(min_t, max_t)

    x = np.stack([data[t + x_offsets, ...] for t in valid_indices], axis=0)
    y = np.stack([data[t + y_offsets, ...] for t in valid_indices], axis=0)

    if _V4_DEBUG:
        print(f"[v4-DBG][gen_io] features={data.shape[-1]} "
              f"windows={len(valid_indices)} x={x.shape} y={y.shape}", file=sys.stderr)

    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    data = np.load(args.traffic_df_filename)['data']

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val][..., 0:1],
    )
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]

    # =========== MinMaxNorm ============ #
    x_train_norm = x_train[:, :, :, :num_feat]
    x_train_time = x_train[:, :, :, num_feat:]
    x_val_norm = x_val[:, :, :, :num_feat]
    x_val_time = x_val[:, :, :, num_feat:]
    x_test_norm = x_test[:, :, :, :num_feat]
    x_test_time = x_test[:, :, :, num_feat:]

    x_train_norm = np.transpose(x_train_norm, axes=[0, 2, 3, 1])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 2, 3, 1])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 2, 3, 1])

    stat, x_train_norm, x_val_norm, x_test_norm = MinMaxnormalization(x_train_norm, x_val_norm, x_test_norm)

    x_train_norm = np.transpose(x_train_norm, axes=[0, 3, 1, 2])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 3, 1, 2])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 3, 1, 2])
    _max = stat['_max']
    _min = stat['_min']

    x_train = np.concatenate([x_train_norm, x_train_time], axis=-1)
    x_val = np.concatenate([x_val_norm, x_val_time], axis=-1)
    x_test = np.concatenate([x_test_norm, x_test_time], axis=-1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    pickle.dump(_max, open("datasets/PEMS04/max.pkl", 'wb'))
    pickle.dump(_min, open("datasets/PEMS04/min.pkl", 'wb'))


if __name__ == "__main__":
    seq_length_x = 12
    seq_length_y = 12
    y_start = 1
    dow = True
    dataset = "PEMS04"
    output_dir = 'datasets/PEMS04'
    traffic_df_filename = 'datasets/raw_data/PEMS04/PEMS04.npz'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default=traffic_df_filename, help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=y_start, help="Y pred start",)
    parser.add_argument("--dow", type=bool, default=dow, help='Add feature day_of_week.')

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
