from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

# Delta vs upstream:
#   1. Split ratios parameterised as CLI args (--train_ratio / --test_ratio)
#   2. Prints per-split NaN counts as sanity check


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    add_one_hot = False
    num_time_slot_a_day = 288
    num_day_a_week      = 7
    print("warning: number of time slot in a day is set to {0}".format(num_time_slot_a_day))
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        if add_one_hot:
            time_in_day_index = list(range(data.shape[0]))
            time_in_day_one_hot_index = [_ % num_time_slot_a_day for _ in time_in_day_index]
            time_in_day_one_hot_index = torch.tensor(time_in_day_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)

    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        if add_one_hot:
            day_in_week_index = list(range(data.shape[0]))
            day_in_week_one_hot_index = [_ % num_day_a_week for _ in day_in_week_index]
            time_in_day_one_hot_index = torch.tensor(day_in_week_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)

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
    df = pd.read_hdf(args.traffic_df_filename)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    num_samples = x.shape[0]
    # ── delta 1: parameterised split ratios ──
    num_test  = round(num_samples * args.test_ratio)
    num_train = round(num_samples * args.train_ratio)
    num_val   = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # ── delta 2: NaN sanity check ──
        nan_x = int(np.isnan(_x).sum())
        nan_y = int(np.isnan(_y).sum())
        print(f"{cat}  x: {_x.shape}  y: {_y.shape}  NaN(x)={nan_x} NaN(y)={nan_y}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    seq_length_x = 12
    seq_length_y = 12
    y_start      = 1
    dow          = True
    output_dir   = 'datasets/PEMS-BAY'
    traffic_df_filename = 'datasets/raw_data/PEMS-BAY/pems-bay.h5'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--traffic_df_filename", type=str, default=traffic_df_filename)
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y)
    parser.add_argument("--y_start", type=int, default=y_start)
    parser.add_argument("--dow", type=bool, default=dow)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--test_ratio",  type=float, default=0.2)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
