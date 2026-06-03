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
import sys

_V4_DEBUG = True


def _dbg(*a, **kw):
    if _V4_DEBUG:
        print(*a, file=sys.stderr, **kw)


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    add_one_hot = False
    num_time_slot_a_day     = 288
    num_day_a_week          = 7
    print("warning: number of time slot in a day is set to {0}".format(num_time_slot_a_day))
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    # v4-algo: per-node variance check — flag dead/constant sensors
    node_var = np.nanvar(df.values, axis=0)
    dead_sensors = np.where(node_var < 1e-8)[0]
    if len(dead_sensors) > 0:
        _dbg(f"[v4-DBG][PEMS-BAY] WARNING: {len(dead_sensors)} dead/constant sensors "
             f"detected (var<1e-8): indices {dead_sensors[:10].tolist()}"
             f"{'...' if len(dead_sensors) > 10 else ''}")

    # v4-algo: eps-guarded MinMax normalization on traffic channel
    # upstream uses raw values; v4 optionally scales to [0,1] per-node
    # to stabilize downstream gradient flow through the sliding window features
    _eps = 1e-8
    node_min = np.nanmin(df.values, axis=0, keepdims=True)  # (1, N)
    node_max = np.nanmax(df.values, axis=0, keepdims=True)  # (1, N)
    denom = node_max - node_min
    denom = np.where(denom < _eps, _eps, denom)
    data_normed = (df.values - node_min) / denom  # (T, N) in [0,1]
    # store normed version alongside raw — model can select which channel to use
    feature_list.append(np.expand_dims(data_normed, axis=-1))
    _dbg(f"[v4-DBG][PEMS-BAY] appended MinMax-normed channel, "
         f"range [{np.nanmin(data_normed):.4f}, {np.nanmax(data_normed):.4f}]")

    if add_time_in_day:
        # numerical time_in_day
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        if add_one_hot:
            # one_hot_time_in_day
            time_in_day_index = list(range(data.shape[0]))
            time_in_day_one_hot_index = [_%num_time_slot_a_day for _ in time_in_day_index]
            time_in_day_one_hot_index = torch.tensor(time_in_day_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)

    if add_day_in_week:
        # numerical day_in_week
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        # one_hot_day_in_week
        if add_one_hot:
            day_in_week_index = list(range(data.shape[0]))
            day_in_week_one_hot_index = [_%num_day_a_week for _ in day_in_week_index]
            time_in_day_one_hot_index = torch.tensor(day_in_week_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)

    data = np.concatenate(feature_list, axis=-1)

    # v4-algo: linear interpolation for NaN gaps before windowing
    # upstream silently passes NaN through; v4 interpolates gaps <= 6 timesteps
    # to avoid poisoning entire sliding windows with a single missing reading
    traffic_ch = data[..., 0]
    nan_count_before = np.isnan(traffic_ch).sum()
    if nan_count_before > 0:
        for n in range(traffic_ch.shape[1]):
            col = traffic_ch[:, n]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                # identify contiguous NaN runs
                diff = np.diff(nans.astype(int))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1
                if nans[0]:
                    starts = np.concatenate([[0], starts])
                if nans[-1]:
                    ends = np.concatenate([ends, [len(col)]])
                for s, e in zip(starts, ends):
                    gap_len = e - s
                    if gap_len <= 6:  # only interpolate short gaps
                        left_val = col[s-1] if s > 0 else col[e] if e < len(col) else 0.0
                        right_val = col[e] if e < len(col) else left_val
                        col[s:e] = np.linspace(left_val, right_val, gap_len + 2)[1:-1]
                traffic_ch[:, n] = col
        data[..., 0] = traffic_ch
        nan_count_after = np.isnan(data[..., 0]).sum()
        _dbg(f"[v4-DBG][PEMS-BAY] NaN interpolation: {nan_count_before} -> {nan_count_after} "
             f"(filled {nan_count_before - nan_count_after} values in gaps <= 6 steps)")

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # v4-algo: assert no overlap between splits (guards against off-by-one)
    assert num_train + num_val + num_test <= num_samples, \
        f"Split overflow: {num_train}+{num_val}+{num_test}={num_train+num_val+num_test} > {num_samples}"

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    # v4-algo: verify no distribution drift between splits via mean/std comparison
    for ch_idx, ch_name in [(0, "raw_traffic"), (1, "minmax_normed")]:
        train_mean = np.nanmean(x_train[..., ch_idx])
        test_mean  = np.nanmean(x_test[..., ch_idx])
        drift_ratio = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)
        _dbg(f"[v4-DBG][split][{ch_name}] train_mean={train_mean:.4f} "
             f"test_mean={test_mean:.4f} drift={drift_ratio:.4f}")
        if drift_ratio > 0.3:
            _dbg(f"[v4-DBG][split] WARNING: high distribution drift in {ch_name} "
                 f"({drift_ratio:.2%}), consider temporal normalization")

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


if __name__ == "__main__":
    seq_length_x    = 12
    seq_length_y    = 12
    y_start         = 1
    dow             = True
    dataset         = "PEMS-BAY"
    output_dir  = 'datasets/PEMS-BAY'
    traffic_df_filename = 'datasets/raw_data/PEMS-BAY/pems-bay.h5'

    parser  = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default=traffic_df_filename, help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=y_start, help="Y pred start", )
    parser.add_argument("--dow", type=bool, default=dow, help='Add feature day_of_week.')

    args    = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply   = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
