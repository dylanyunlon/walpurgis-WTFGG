"""
Generate train/val/test splits for traffic FLOW datasets (PEMS04, PEMS08).
Uses min-max normalization on raw traffic features, unlike the speed
datasets (METR-LA, PEMS-BAY) which use z-score.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import pickle
import sys

import numpy as np

_DBG_GEN = ("--debug-gen" in sys.argv) or False
_SLOTS_PER_DAY = 288
_N_FEAT = 1


def _minmax_normalize(train, val, test):
    """
    Min-max normalize traffic features to [-1, 1].
    Arrays shaped [B, N, F, T].
    Returns (stats_dict, normed_train, normed_val, normed_test).
    """
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    if _DBG_GEN:
        print(f"[DBG:gen_flow] minmax  _max={_max.squeeze()}  _min={_min.squeeze()}")

    def norm(x):
        scaled = (x - _min) / (_max - _min + 1e-12)
        return scaled * 2.0 - 1.0

    return {'_max': _max, '_min': _min}, norm(train), norm(val), norm(test)


def build_flow_samples(data, x_offsets, y_offsets,
                       use_tod=True, use_dow=True):
    """
    Slide a window over the npz data array [T, N, F].
    Returns x [samples, L_in, N, D], y [samples, L_out, N, D].
    """
    n_timestamps, n_nodes, _ = data.shape
    features = [data[..., :_N_FEAT]]

    if use_tod:
        tod = np.array([i % _SLOTS_PER_DAY / _SLOTS_PER_DAY for i in range(n_timestamps)])
        tod = np.tile(tod, [1, n_nodes, 1]).transpose((2, 1, 0))
        features.append(tod)

    if use_dow:
        dow = np.array([(i // _SLOTS_PER_DAY) % 7 for i in range(n_timestamps)])
        dow = np.tile(dow, [1, n_nodes, 1]).transpose((2, 1, 0))
        features.append(dow)

    combined = np.concatenate(features, axis=-1)
    xs, ys = [], []
    lo = abs(min(x_offsets))
    hi = n_timestamps - abs(max(y_offsets))
    for t in range(lo, hi):
        xs.append(combined[t + x_offsets])
        ys.append(combined[t + y_offsets])

    return np.stack(xs, 0), np.stack(ys, 0)


def split_and_save_flow(args):
    """Load .npz, build samples, normalize, split 60/20/20, save."""
    raw = np.load(args.traffic_df_filename)['data']
    seq_x, seq_y = args.seq_length_x, args.seq_length_y

    x_offsets = np.sort(np.arange(-(seq_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(args.y_start, seq_y + 1, 1))

    x, y = build_flow_samples(raw, x_offsets, y_offsets,
                               use_tod=True, use_dow=args.dow)
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    n = x.shape[0]
    n_test  = round(n * 0.2)
    n_train = round(n * 0.6)
    n_val   = n - n_test - n_train

    x_train, y_train = x[:n_train],                       y[:n_train][..., :1]
    x_val,   y_val   = x[n_train:n_train+n_val],          y[n_train:n_train+n_val][..., :1]
    x_test,  y_test  = x[-n_test:],                        y[-n_test:][..., :1]

    # ── min-max normalization on traffic features only ──
    xtr_feat = np.transpose(x_train[..., :_N_FEAT], [0, 2, 3, 1])
    xva_feat = np.transpose(x_val  [..., :_N_FEAT], [0, 2, 3, 1])
    xte_feat = np.transpose(x_test [..., :_N_FEAT], [0, 2, 3, 1])

    stats, xtr_n, xva_n, xte_n = _minmax_normalize(xtr_feat, xva_feat, xte_feat)

    xtr_n = np.transpose(xtr_n, [0, 3, 1, 2])
    xva_n = np.transpose(xva_n, [0, 3, 1, 2])
    xte_n = np.transpose(xte_n, [0, 3, 1, 2])

    x_train = np.concatenate([xtr_n, x_train[..., _N_FEAT:]], axis=-1)
    x_val   = np.concatenate([xva_n, x_val  [..., _N_FEAT:]], axis=-1)
    x_test  = np.concatenate([xte_n, x_test [..., _N_FEAT:]], axis=-1)

    # ── save splits ──
    os.makedirs(args.output_dir, exist_ok=True)
    for name, sx, sy in [('train', x_train, y_train),
                          ('val',   x_val,   y_val),
                          ('test',  x_test,  y_test)]:
        print(f"  {name}  x={sx.shape}  y={sy.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{name}.npz"),
            x=sx, y=sy,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    # save normalization stats
    pickle.dump(stats['_max'], open(os.path.join(args.output_dir, "max.pkl"), 'wb'))
    pickle.dump(stats['_min'], open(os.path.join(args.output_dir, "min.pkl"), 'wb'))
    print("Done.")


def make_flow_parser(dataset, output_dir, npz_path):
    p = argparse.ArgumentParser(description=f"Generate training data for {dataset}")
    p.add_argument("--output_dir",          type=str, default=output_dir)
    p.add_argument("--traffic_df_filename", type=str, default=npz_path)
    p.add_argument("--seq_length_x",        type=int, default=12)
    p.add_argument("--seq_length_y",        type=int, default=12)
    p.add_argument("--y_start",             type=int, default=1)
    p.add_argument("--dow",                 type=bool, default=True)
    return p
