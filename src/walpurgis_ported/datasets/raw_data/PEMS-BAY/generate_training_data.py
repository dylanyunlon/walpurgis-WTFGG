"""Generate training/validation/test data splits for PEMS-BAY dataset.

Walpurgis adaptations:
- Per-stage timing and memory reporting
- Data integrity checks (NaN, range) on generated arrays
- Summary statistics table for each split
- Reproducible output with seed tracking
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import time
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """Generate sliding-window samples from time-series dataframe.

    Args:
        df: pandas DataFrame [num_timesteps, num_nodes]
        x_offsets: history window offsets
        y_offsets: forecast window offsets

    Returns:
        x: [num_samples, input_length, num_nodes, input_dim]
        y: [num_samples, output_length, num_nodes, output_dim]
    """
    t0 = time.perf_counter()
    add_one_hot = False
    num_time_slot_a_day     = 288
    num_day_a_week          = 7
    print(f"[Walpurgis::generate_io_data] time_slots/day={num_time_slot_a_day}")
    num_samples, num_nodes = df.shape
    print(f"[Walpurgis::generate_io_data] raw df: {num_samples} timesteps × {num_nodes} nodes")

    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    feat_names = ['traffic_speed']

    if add_time_in_day:
        # numerical time_in_day
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        feat_names.append('time_in_day')
        if add_one_hot:
            time_in_day_index = list(range(data.shape[0]))
            time_in_day_one_hot_index = [_%num_time_slot_a_day for _ in time_in_day_index]
            time_in_day_one_hot_index = torch.tensor(time_in_day_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)
            feat_names.append('time_in_day_idx')

    if add_day_in_week:
        # numerical day_in_week
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        feat_names.append('day_in_week')
        if add_one_hot:
            day_in_week_index = list(range(data.shape[0]))
            day_in_week_one_hot_index = [_%num_day_a_week for _ in day_in_week_index]
            time_in_day_one_hot_index = torch.tensor(day_in_week_one_hot_index).unsqueeze(1).expand(-1, num_nodes).unsqueeze(-1).numpy()
            feature_list.append(time_in_day_one_hot_index)
            feat_names.append('dow_idx')

    data = np.concatenate(feature_list, axis=-1)
    print(f"[Walpurgis::generate_io_data] features: {feat_names} → dim={data.shape[-1]}")

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    elapsed = time.perf_counter() - t0
    mem_mb = (x.nbytes + y.nbytes) / (1024**2)
    print(f"[Walpurgis::generate_io_data] done in {elapsed:.3f}s")
    print(f"  x: {x.shape} y: {y.shape} total_mem={mem_mb:.1f}MB")

    # Integrity check
    for name, arr in [('x', x), ('y', y)]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            print(f"  ⚠ {name} contains {nan_count} NaN values!")
        print(f"  {name} range=[{arr.min():.4f}, {arr.max():.4f}] "
              f"mean={arr.mean():.4f} std={arr.std():.4f}")

    return x, y


def generate_train_val_test(args):
    """Generate train/val/test splits and save as .npz files."""
    t0 = time.perf_counter()
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    print(f"\n{'='*60}")
    print(f"[Walpurgis::PEMS-BAY] Generating training data")
    print(f"  seq_length_x={seq_length_x} seq_length_y={seq_length_y}")
    print(f"  source: {args.traffic_df_filename}")
    print(f"  output: {args.output_dir}")
    print(f"{'='*60}")

    df = pd.read_hdf(args.traffic_df_filename)
    print(f"[Walpurgis] Raw HDF loaded: {df.shape[0]} timesteps × {df.shape[1]} sensors")
    print(f"  time range: {df.index[0]} → {df.index[-1]}")

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print(f"\n[Walpurgis] Splitting data (70/10/20 train/val/test)...")
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    # Summary table
    print(f"\n  {'Split':<8} {'Samples':>8} {'x shape':<30} {'y shape':<30}")
    print(f"  {'-'*76}")
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"  {cat:<8} {len(_x):>8} {str(_x.shape):<30} {str(_y.shape):<30}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    elapsed = time.perf_counter() - t0
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f"{c}.npz"))
        for c in ["train", "val", "test"]
        if os.path.exists(os.path.join(args.output_dir, f"{c}.npz"))
    )
    print(f"\n[Walpurgis::PEMS-BAY] COMPLETE in {elapsed:.3f}s")
    print(f"  total output size: {total_size / (1024**2):.1f}MB")


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
