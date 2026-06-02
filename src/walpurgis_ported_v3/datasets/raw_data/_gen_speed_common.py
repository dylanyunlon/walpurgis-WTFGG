"""
Common generator for speed-type datasets (METR-LA, PEMS-BAY).
Shared logic extracted to reduce duplication across dataset folders.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import pandas as pd


def _build_seq2seq_samples(df, x_off, y_off, add_tod=True, add_dow=True):
    """Slide window over dataframe to produce (x, y) sample pairs."""
    SLOTS_PER_DAY = 288
    print(f"[gen] time slots/day = {SLOTS_PER_DAY}")
    n_samples, n_nodes = df.shape
    raw = np.expand_dims(df.values, axis=-1)
    feats = [raw]

    if add_tod:
        frac = (df.index.values - df.index.values.astype("datetime64[D]")) \
               / np.timedelta64(1, "D")
        tod = np.tile(frac, [1, n_nodes, 1]).transpose((2, 1, 0))
        feats.append(tod)

    if add_dow:
        dow_arr = df.index.dayofweek
        dow_t = np.tile(dow_arr, [1, n_nodes, 1]).transpose((2, 1, 0))
        feats.append(dow_t)

    data = np.concatenate(feats, axis=-1)
    xs, ys = [], []
    lo = abs(min(x_off))
    hi = abs(n_samples - abs(max(y_off)))
    for t in range(lo, hi):
        xs.append(data[t + x_off, ...])
        ys.append(data[t + y_off, ...])
    return np.stack(xs, 0), np.stack(ys, 0)


def run_speed_gen(dataset_tag, output_dir, h5_path,
                  seq_x=12, seq_y=12, y_start=1, dow=True):
    """Full pipeline: read h5 -> split -> save npz."""
    df = pd.read_hdf(h5_path)
    x_off = np.sort(np.concatenate((np.arange(-(seq_x - 1), 1, 1),)))
    y_off = np.sort(np.arange(y_start, seq_y + 1, 1))

    x, y = _build_seq2seq_samples(df, x_off, y_off, add_tod=True, add_dow=dow)
    print(f"[gen] x={x.shape}  y={y.shape}")

    N = x.shape[0]
    n_test  = round(N * 0.2)
    n_train = round(N * 0.7)
    n_val   = N - n_test - n_train

    splits = {
        'train': (x[:n_train],              y[:n_train]),
        'val':   (x[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        'test':  (x[-n_test:],              y[-n_test:]),
    }
    os.makedirs(output_dir, exist_ok=True)
    for name, (_x, _y) in splits.items():
        print(f"  {name}  x={_x.shape}  y={_y.shape}")
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            x=_x, y=_y,
            x_offsets=x_off.reshape(list(x_off.shape) + [1]),
            y_offsets=y_off.reshape(list(y_off.shape) + [1]),
        )
    print(f"[gen] {dataset_tag} done -> {output_dir}")
