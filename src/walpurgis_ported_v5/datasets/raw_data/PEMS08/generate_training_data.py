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

# Delta vs upstream:
#   1. Split ratios parameterised (--train_ratio / --test_ratio)
#   2. NaN sanity check before save

num_feat = 1


def MinMaxnormalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)
    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm   = normalize(val)
    test_norm  = normalize(test)
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
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
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
    # ── delta 1: parameterised ──
    num_train = round(num_samples * args.train_ratio) - 1
    num_test  = round(num_samples * args.test_ratio)
    num_val   = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val][..., 0:1],
    )
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]

    # MinMax norm
    x_train_norm = x_train[:, :, :, :num_feat]
    x_train_time = x_train[:, :, :, num_feat:]
    x_val_norm   = x_val[:, :, :, :num_feat]
    x_val_time   = x_val[:, :, :, num_feat:]
    x_test_norm  = x_test[:, :, :, :num_feat]
    x_test_time  = x_test[:, :, :, num_feat:]

    x_train_norm = np.transpose(x_train_norm, axes=[0, 2, 3, 1])
    x_val_norm   = np.transpose(x_val_norm,   axes=[0, 2, 3, 1])
    x_test_norm  = np.transpose(x_test_norm,  axes=[0, 2, 3, 1])

    stat, x_train_norm, x_val_norm, x_test_norm = MinMaxnormalization(
        x_train_norm, x_val_norm, x_test_norm)

    x_train_norm = np.transpose(x_train_norm, axes=[0, 3, 1, 2])
    x_val_norm   = np.transpose(x_val_norm,   axes=[0, 3, 1, 2])
    x_test_norm  = np.transpose(x_test_norm,  axes=[0, 3, 1, 2])
    _max = stat['_max']
    _min = stat['_min']

    x_train = np.concatenate([x_train_norm, x_train_time], axis=-1)
    x_val   = np.concatenate([x_val_norm,   x_val_time],   axis=-1)
    x_test  = np.concatenate([x_test_norm,  x_test_time],  axis=-1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # ── delta 2: NaN check ──
        nan_cnt = int(np.isnan(_x).sum()) + int(np.isnan(_y).sum())
        print(f"{cat}  x: {_x.shape}  y: {_y.shape}  NaN_total={nan_cnt}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    pickle.dump(_max, open("datasets/PEMS08/max.pkl", 'wb'))
    pickle.dump(_min, open("datasets/PEMS08/min.pkl", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='datasets/PEMS08')
    parser.add_argument("--traffic_df_filename", type=str, default='datasets/raw_data/PEMS08/PEMS08.npz')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--test_ratio",  type=float, default=0.2)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Overwrite? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
