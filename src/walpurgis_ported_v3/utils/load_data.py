"""
Dataset loading: normalization, scaler, adjacency deserialization.
Ported with debug probes on every data path.
"""
import sys
import os
import pickle
import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *

_DBG = ("--debug-data" in sys.argv)

# ──────────── normalization helpers ────────────

def re_normalization(x, mu, sigma):
    """Inverse of z-score: x * σ + μ"""
    return x * sigma + mu


def max_min_normalization(x, hi, lo):
    """Scale x into [-1, 1] given observed min/max."""
    x = (x - lo) / (hi - lo)           # -> [0, 1]
    return x * 2.0 - 1.0                # -> [-1, 1]


def re_max_min_normalization(x, hi, lo):
    """Inverse of max-min norm: [-1,1] -> original range."""
    x = (x + 1.0) * 0.5                # -> [0, 1]
    return x * (hi - lo) + lo


# ──────────── StandardScaler ────────────

class StandardScaler:
    """Zero-mean / unit-variance normalizer, training-set statistics."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if _DBG:
            print(f"[DBG:data] StandardScaler  mean={mean:.6f}  std={std:.6f}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ──────────── pickle loader ────────────

def load_pickle(fpath):
    try:
        with open(fpath, 'rb') as fh:
            obj = pickle.load(fh)
    except UnicodeDecodeError:
        with open(fpath, 'rb') as fh:
            obj = pickle.load(fh, encoding='latin1')
    except Exception as exc:
        print(f'Cannot deserialize {fpath}: {exc}')
        raise
    if _DBG:
        tp = type(obj).__name__
        sz = os.path.getsize(fpath)
        print(f"[DBG:data] load_pickle  {fpath}  type={tp}  "
              f"file_bytes={sz}")
    return obj


# ──────────── main loader ────────────

def load_dataset(data_dir, bs_train, bs_val, bs_test, ds_name):
    """Read pre-generated train/val/test npz and build DataLoaders."""
    bucket = {}

    for split in ['train', 'val', 'test']:
        npz = np.load(os.path.join(data_dir, f'{split}.npz'))
        bucket[f'x_{split}'] = npz['x']
        bucket[f'y_{split}'] = npz['y']
        if _DBG:
            print(f"[DBG:data] loaded {split}  "
                  f"x={npz['x'].shape}  y={npz['y'].shape}")

    is_flow = ds_name in ('PEMS04', 'PEMS08')

    if is_flow:
        # ---- traffic flow: min-max normalization ----
        _min = pickle.load(open(f"datasets/{ds_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{ds_name}/max.pkl", 'rb'))
        if _DBG:
            print(f"[DBG:data] flow dataset  _min.shape={np.asarray(_min).shape}  "
                  f"_max.shape={np.asarray(_max).shape}")

        for split in ['train', 'val', 'test']:
            y_raw = bucket[f'y_{split}']
            y_t = np.squeeze(np.transpose(y_raw, axes=[0, 2, 1, 3]), axis=-1)
            y_normed = max_min_normalization(y_t, _max[:, :, 0, :], _min[:, :, 0, :])
            bucket[f'y_{split}'] = np.transpose(y_normed, axes=[0, 2, 1])

        bucket['train_loader'] = DataLoader(bucket['x_train'], bucket['y_train'], bs_train, shuffle=True)
        bucket['val_loader']   = DataLoader(bucket['x_val'],   bucket['y_val'],   bs_val)
        bucket['test_loader']  = DataLoader(bucket['x_test'],  bucket['y_test'],  bs_test)
        bucket['scaler']       = re_max_min_normalization

    else:
        # ---- traffic speed: z-score ----
        scaler = StandardScaler(
            mean=bucket['x_train'][..., 0].mean(),
            std=bucket['x_train'][..., 0].std()
        )
        for split in ['train', 'val', 'test']:
            bucket[f'x_{split}'][..., 0] = scaler.transform(bucket[f'x_{split}'][..., 0])
            bucket[f'y_{split}'][..., 0] = scaler.transform(bucket[f'y_{split}'][..., 0])

        bucket['train_loader'] = DataLoader(bucket['x_train'], bucket['y_train'], bs_train, shuffle=True)
        bucket['val_loader']   = DataLoader(bucket['x_val'],   bucket['y_val'],   bs_val)
        bucket['test_loader']  = DataLoader(bucket['x_test'],  bucket['y_test'],  bs_test)
        bucket['scaler']       = scaler

    if _DBG:
        print(f"[DBG:data] load_dataset complete  "
              f"train_batches={len(bucket['train_loader'])}  "
              f"val_batches={len(bucket['val_loader'])}  "
              f"test_batches={len(bucket['test_loader'])}")

    return bucket


# ──────────── adjacency loader ────────────

def load_adj(file_path, adj_type):
    """Deserialize raw adj and compute the requested transform."""
    try:
        _ids, _id2idx, adj_raw = load_pickle(file_path)
    except (ValueError, TypeError):
        adj_raw = load_pickle(file_path)

    if _DBG:
        print(f"[DBG:data] load_adj  path={file_path}  "
              f"adj_type={adj_type}  shape={np.asarray(adj_raw).shape}")

    dispatch = {
        "scalap":          lambda m: [calculate_scaled_laplacian(m).astype(np.float32).todense()],
        "normlap":         lambda m: [calculate_symmetric_normalized_laplacian(m).astype(np.float32).todense()],
        "symnadj":         lambda m: [symmetric_message_passing_adj(m).astype(np.float32).todense()],
        "transition":      lambda m: [transition_matrix(m).T],
        "doubletransition": lambda m: [transition_matrix(m).T, transition_matrix(m.T).T],
        "identity":        lambda m: [np.diag(np.ones(m.shape[0])).astype(np.float32).todense()],
        "original":        lambda m: m,
    }

    if adj_type not in dispatch:
        raise ValueError(f"Unknown adj_type '{adj_type}'")

    processed = dispatch[adj_type](adj_raw)

    if _DBG:
        print(f"[DBG:data] load_adj result  "
              f"n_matrices={len(processed) if isinstance(processed, list) else 'raw'}")

    return processed, adj_raw
