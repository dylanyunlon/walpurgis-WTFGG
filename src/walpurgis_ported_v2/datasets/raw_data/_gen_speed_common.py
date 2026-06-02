"""
Generate train/val/test splits from raw HDF5 traffic data.
Shared logic for METR-LA and PEMS-BAY (speed datasets).
Each dataset's folder has a thin wrapper that sets defaults.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import numpy as np
import pandas as pd

_DBG_GEN = ("--debug-gen" in sys.argv) or False
_SLOTS_PER_DAY = 288
_DAYS_PER_WEEK = 7


def build_seq2seq_samples(df, x_offsets, y_offsets,
                          use_time_of_day=True, use_day_of_week=True):
    """
    Slide a window over the dataframe and produce (x, y) sample arrays.

    Returns
    -------
    x : [N_samples, len(x_offsets), N_nodes, D]
    y : [N_samples, len(y_offsets), N_nodes, D]
    """
    n_timestamps, n_nodes = df.shape
    raw = np.expand_dims(df.values, axis=-1)
    features = [raw]

    if use_time_of_day:
        frac = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod = np.tile(frac, [1, n_nodes, 1]).transpose((2, 1, 0))
        features.append(tod)

    if use_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, n_nodes, 1]).transpose((2, 1, 0))
        features.append(dow_tiled)

    data = np.concatenate(features, axis=-1)

    if _DBG_GEN:
        print(f"[DBG:gen] data assembled  shape={data.shape}  "
              f"features={'|'.join(['raw','tod','dow'][:len(features)])}")

    xs, ys = [], []
    lo = abs(min(x_offsets))
    hi = n_timestamps - abs(max(y_offsets))
    for t in range(lo, hi):
        xs.append(data[t + x_offsets, ...])
        ys.append(data[t + y_offsets, ...])

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def split_and_save(args):
    """Load HDF5, build samples, split 70/10/20, save as .npz."""
    df = pd.read_hdf(args.traffic_df_filename)
    seq_x, seq_y = args.seq_length_x, args.seq_length_y

    x_offsets = np.sort(np.arange(-(seq_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(args.y_start, seq_y + 1, 1))

    x, y = build_seq2seq_samples(
        df, x_offsets, y_offsets,
        use_time_of_day=True, use_day_of_week=args.dow,
    )
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    n = x.shape[0]
    n_test  = round(n * 0.2)
    n_train = round(n * 0.7)
    n_val   = n - n_test - n_train

    splits = {
        'train': (x[:n_train],                       y[:n_train]),
        'val':   (x[n_train:n_train+n_val],          y[n_train:n_train+n_val]),
        'test':  (x[-n_test:],                       y[-n_test:]),
    }

    for name, (sx, sy) in splits.items():
        print(f"  {name}  x={sx.shape}  y={sy.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{name}.npz"),
            x=sx, y=sy,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    print("Done.")


def make_parser(dataset, output_dir, h5_path):
    """Build argparse with dataset-specific defaults."""
    p = argparse.ArgumentParser(description=f"Generate training data for {dataset}")
    p.add_argument("--output_dir",            type=str, default=output_dir)
    p.add_argument("--traffic_df_filename",   type=str, default=h5_path)
    p.add_argument("--seq_length_x",          type=int, default=12)
    p.add_argument("--seq_length_y",          type=int, default=12)
    p.add_argument("--y_start",               type=int, default=1)
    p.add_argument("--dow",                   type=bool, default=True)
    return p
