from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import numpy as np
import os
import pandas as pd


def _cyclic_time_encoding(time_ind, period):
    angle = 2.0 * np.pi * time_ind / period
    return np.sin(angle), np.cos(angle)


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True,
        stride=1):
    num_time_slot_a_day = 288
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        time_ind = ((df.index.values - df.index.values.astype("datetime64[D]"))
                    / np.timedelta64(1, "D"))
        time_linear = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_linear)

    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    # stride 跳步采样
    for t in range(min_t, max_t, stride):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x = args.seq_length_x
    seq_length_y = args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)

    x_offsets = np.sort(np.concatenate(
        (np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow,
        stride=args.stride)

    print(f"x shape: {x.shape}, y shape: {y.shape}")

    num_samples = x.shape[0]
    samples_per_week = 288 * 7
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_test = (num_test // samples_per_week) * samples_per_week or round(num_samples * 0.2)
    num_train = (num_train // samples_per_week) * samples_per_week or round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (x[num_train:num_train + num_val],
                    y[num_train:num_train + num_val])
    x_test, y_test = x[-num_test:], y[-num_test:]

    stats = {
        'total_samples': int(num_samples),
        'train': int(num_train), 'val': int(num_val), 'test': int(num_test),
        'num_nodes': int(x.shape[2]), 'feat_dim': int(x.shape[3]),
        'x_mean': float(x_train[..., 0].mean()),
        'x_std': float(x_train[..., 0].std()),
        'x_min': float(x_train[..., 0].min()),
        'x_max': float(x_train[..., 0].max()),
        'stride': args.stride,
    }

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{cat}: x={_x.shape}, y={_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    with open(os.path.join(args.output_dir, 'data_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[v10] Stats saved: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default='datasets/PEMS-BAY')
    parser.add_argument("--traffic_df_filename", type=str,
                        default='datasets/raw_data/PEMS-BAY/pems-bay.h5')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(
            f'{args.output_dir} exists. Overwrite? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
