"""
Common generator for flow-type datasets (PEMS04, PEMS08).
Includes MinMax normalization pipeline.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pickle
import numpy as np
import os

NUM_FEAT = 1


def _minmax_norm(train, val, test):
    """Compute min-max stats on train, apply to all splits."""
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)
    print(f"  _max.shape={_max.shape}  _min.shape={_min.shape}")

    def _scale(x):
        return 2.0 * (x - _min) / (_max - _min) - 1.0

    return {'_max': _max, '_min': _min}, _scale(train), _scale(val), _scale(test)


def _build_flow_samples(data, x_off, y_off, add_tod=True, add_dow=True):
    """Slide window over (T, N, F) array."""
    n_samples, n_nodes, _ = data.shape
    feats = [data[..., 0:NUM_FEAT]]

    if add_tod:
        tod = np.array([i % 288 / 288 for i in range(n_samples)])
        tod = np.tile(tod, [1, n_nodes, 1]).transpose((2, 1, 0))
        feats.append(tod)

    if add_dow:
        dow = np.array([(i // 288) % 7 for i in range(n_samples)])
        dow = np.tile(dow, [1, n_nodes, 1]).transpose((2, 1, 0))
        feats.append(dow)

    data = np.concatenate(feats, axis=-1)
    xs, ys = [], []
    lo = abs(min(x_off))
    hi = abs(n_samples - abs(max(y_off)))
    for t in range(lo, hi):
        xs.append(data[t + x_off, ...])
        ys.append(data[t + y_off, ...])
    return np.stack(xs, 0), np.stack(ys, 0)


def run_flow_gen(dataset_tag, output_dir, npz_path,
                 train_ratio=0.6, seq_x=12, seq_y=12,
                 y_start=1, dow=True, train_minus_one=False):
    """Full pipeline: read npz -> split -> minmax -> save."""

    data = np.load(npz_path)['data']
    x_off = np.sort(np.concatenate((np.arange(-(seq_x - 1), 1, 1),)))
    y_off = np.sort(np.arange(y_start, seq_y + 1, 1))

    x, y = _build_flow_samples(data, x_off, y_off,
                                add_tod=True, add_dow=dow)
    print(f"[gen] x={x.shape}  y={y.shape}")

    N = x.shape[0]
    n_test  = round(N * 0.2)
    n_train = round(N * train_ratio)
    if train_minus_one:
        n_train -= 1
    n_val = N - n_test - n_train

    x_tr, y_tr = x[:n_train],               y[:n_train][..., 0:1]
    x_va, y_va = x[n_train:n_train+n_val],  y[n_train:n_train+n_val][..., 0:1]
    x_te, y_te = x[-n_test:],               y[-n_test:][..., 0:1]

    # separate signal vs time features, normalize signal only
    def _norm_split(xp):
        return xp[:, :, :, :NUM_FEAT], xp[:, :, :, NUM_FEAT:]

    xtr_s, xtr_t = _norm_split(x_tr)
    xva_s, xva_t = _norm_split(x_va)
    xte_s, xte_t = _norm_split(x_te)

    xtr_s = np.transpose(xtr_s, [0, 2, 3, 1])
    xva_s = np.transpose(xva_s, [0, 2, 3, 1])
    xte_s = np.transpose(xte_s, [0, 2, 3, 1])

    stat, xtr_s, xva_s, xte_s = _minmax_norm(xtr_s, xva_s, xte_s)

    xtr_s = np.transpose(xtr_s, [0, 3, 1, 2])
    xva_s = np.transpose(xva_s, [0, 3, 1, 2])
    xte_s = np.transpose(xte_s, [0, 3, 1, 2])

    x_tr = np.concatenate([xtr_s, xtr_t], axis=-1)
    x_va = np.concatenate([xva_s, xva_t], axis=-1)
    x_te = np.concatenate([xte_s, xte_t], axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    for name, _x, _y in [('train', x_tr, y_tr),
                          ('val', x_va, y_va),
                          ('test', x_te, y_te)]:
        print(f"  {name}  x={_x.shape}  y={_y.shape}")
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            x=_x, y=_y,
            x_offsets=x_off.reshape(list(x_off.shape) + [1]),
            y_offsets=y_off.reshape(list(y_off.shape) + [1]),
        )

    pickle.dump(stat['_max'], open(f"datasets/{dataset_tag}/max.pkl", 'wb'))
    pickle.dump(stat['_min'], open(f"datasets/{dataset_tag}/min.pkl", 'wb'))
    print(f"[gen] {dataset_tag} done → {output_dir}")
