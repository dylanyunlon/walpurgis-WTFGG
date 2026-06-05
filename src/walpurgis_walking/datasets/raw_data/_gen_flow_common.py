"""
Common generator for flow-type datasets (PEMS04, PEMS08).
Handles MinMax normalization specific to traffic flow data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import numpy as np
import os

NUM_FEAT = 1


def _minmax_norm(train, val, test):
    """Per-feature min-max to [-1, 1], computed on training set only."""
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)
    print(f"[gen] _max={_max.shape}  _min={_min.shape}")

    def _norm(x):
        return 2.0 * (x - _min) / (_max - _min) - 1.0

    return {'_max': _max, '_min': _min}, _norm(train), _norm(val), _norm(test)


def _build_flow_samples(data, x_off, y_off, add_tod=True, add_dow=True):
    """Slide window for npz-based flow datasets."""
    n_samples, n_nodes, _ = data.shape
    feats = [data[..., 0:NUM_FEAT]]

    if add_tod:
        tod = np.array([i % 288 / 288.0 for i in range(n_samples)])
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
                 seq_x=12, seq_y=12, y_start=1, dow=True,
                 train_ratio=0.6):
    """Full pipeline: read npz -> split -> minmax -> save."""
    raw = np.load(npz_path)['data']
    x_off = np.sort(np.concatenate((np.arange(-(seq_x - 1), 1, 1),)))
    y_off = np.sort(np.arange(y_start, seq_y + 1, 1))

    x, y = _build_flow_samples(raw, x_off, y_off, add_tod=True, add_dow=dow)
    print(f"[gen] x={x.shape}  y={y.shape}")

    N = x.shape[0]
    n_test  = round(N * 0.2)
    n_train = round(N * train_ratio)
    if dataset_tag == 'PEMS08':
        n_train -= 1                     # upstream quirk preserved
    n_val = N - n_test - n_train

    x_tr, y_tr = x[:n_train],              y[:n_train][..., 0:1]
    x_va, y_va = x[n_train:n_train+n_val], y[n_train:n_train+n_val][..., 0:1]
    x_te, y_te = x[-n_test:],              y[-n_test:][..., 0:1]

    # separate signal vs time features, then normalize signal
    def _split_norm(xa, xb, xc):
        a_sig, a_t = xa[..., :NUM_FEAT], xa[..., NUM_FEAT:]
        b_sig, b_t = xb[..., :NUM_FEAT], xb[..., NUM_FEAT:]
        c_sig, c_t = xc[..., :NUM_FEAT], xc[..., NUM_FEAT:]

        a_sig = np.transpose(a_sig, [0, 2, 3, 1])
        b_sig = np.transpose(b_sig, [0, 2, 3, 1])
        c_sig = np.transpose(c_sig, [0, 2, 3, 1])

        stat, a_n, b_n, c_n = _minmax_norm(a_sig, b_sig, c_sig)

        a_n = np.transpose(a_n, [0, 3, 1, 2])
        b_n = np.transpose(b_n, [0, 3, 1, 2])
        c_n = np.transpose(c_n, [0, 3, 1, 2])

        return (np.concatenate([a_n, a_t], -1),
                np.concatenate([b_n, b_t], -1),
                np.concatenate([c_n, c_t], -1),
                stat)

    x_tr, x_va, x_te, stat = _split_norm(x_tr, x_va, x_te)

    os.makedirs(output_dir, exist_ok=True)
    for name, _x, _y in [('train', x_tr, y_tr),
                          ('val',   x_va, y_va),
                          ('test',  x_te, y_te)]:
        print(f"  {name}  x={_x.shape}  y={_y.shape}")
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            x=_x, y=_y,
            x_offsets=x_off.reshape(list(x_off.shape) + [1]),
            y_offsets=y_off.reshape(list(y_off.shape) + [1]),
        )

    pickle.dump(stat['_max'], open(f"datasets/{dataset_tag}/max.pkl", 'wb'))
    pickle.dump(stat['_min'], open(f"datasets/{dataset_tag}/min.pkl", 'wb'))
    print(f"[gen] {dataset_tag} done -> {output_dir}")
