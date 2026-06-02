#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset loading, normalization, and adjacent matrix preparation.
"""

import pickle
import os
import sys
import numpy as np

from dataloader import DataLoader
from utils.cal_adj import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    symmetric_message_passing_adj,
    transition_matrix,
)

_DBG_DATA = ("--debug-data" in sys.argv) or False

# ───────────────── Normalization helpers ─────────────────


def re_normalization(x, mean, std):
    """Inverse of z-score: x_orig = x * std + mean."""
    return x * std + mean


def max_min_normalization(x, upper, lower):
    """Scale into [-1, 1] via min-max then affine shift."""
    x_norm = (x - lower) / (upper - lower + 1e-12)
    return x_norm * 2.0 - 1.0


def re_max_min_normalization(x, upper, lower):
    """Inverse of max_min_normalization."""
    x_01 = (x + 1.0) / 2.0
    return x_01 * (upper - lower) + lower


class StandardScaler:
    """Z-score scaler: transform(x) = (x - μ) / σ."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if _DBG_DATA:
            print(f"[DBG:load_data] StandardScaler  mean={mean:.6f}  std={std:.6f}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ───────────────────── Pickle I/O ─────────────────────────

def load_pickle(fpath):
    """Robustly load a pickle file (handles Python 2 encoding)."""
    try:
        with open(fpath, 'rb') as fh:
            obj = pickle.load(fh)
    except UnicodeDecodeError:
        with open(fpath, 'rb') as fh:
            obj = pickle.load(fh, encoding='latin1')
    except Exception as exc:
        print(f'Unable to load {fpath}: {exc}')
        raise
    if _DBG_DATA:
        print(f"[DBG:load_data] load_pickle  path={fpath}  type={type(obj).__name__}")
    return obj


# ───────────────── Dataset loading ────────────────────────

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """
    Build train / val / test DataLoaders from preprocessed .npz files.

    Returns a dict with keys:
        x_train, y_train, ..., train_loader, val_loader, test_loader, scaler
    """
    data = {}
    for split in ('train', 'val', 'test'):
        npz = np.load(os.path.join(data_dir, f'{split}.npz'))
        data[f'x_{split}'] = npz['x']
        data[f'y_{split}'] = npz['y']
        if _DBG_DATA:
            print(f"[DBG:load_data] loaded {split}  x={npz['x'].shape}  y={npz['y'].shape}")

    is_flow_dataset = dataset_name in ('PEMS04', 'PEMS08')

    if is_flow_dataset:
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))

        for split in ('train', 'val', 'test'):
            y_raw = np.squeeze(
                np.transpose(data[f'y_{split}'], axes=[0, 2, 1, 3]), axis=-1
            )
            y_normed = max_min_normalization(y_raw, _max[:, :, 0, :], _min[:, :, 0, :])
            data[f'y_{split}'] = np.transpose(y_normed, axes=[0, 2, 1])
            if _DBG_DATA:
                print(f"[DBG:load_data] flow normalization {split}  "
                      f"y_range=[{y_normed.min():.4f}, {y_normed.max():.4f}]")

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader']   = DataLoader(data['x_val'],   data['y_val'],   valid_batch_size)
        data['test_loader']  = DataLoader(data['x_test'],  data['y_test'],  test_batch_size)
        data['scaler'] = re_max_min_normalization

    else:  # traffic speed datasets
        mu  = data['x_train'][..., 0].mean()
        sig = data['x_train'][..., 0].std()
        scaler = StandardScaler(mean=mu, std=sig)

        for split in ('train', 'val', 'test'):
            data[f'x_{split}'][..., 0] = scaler.transform(data[f'x_{split}'][..., 0])
            data[f'y_{split}'][..., 0] = scaler.transform(data[f'y_{split}'][..., 0])

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader']   = DataLoader(data['x_val'],   data['y_val'],   valid_batch_size)
        data['test_loader']  = DataLoader(data['x_test'],  data['y_test'],  test_batch_size)
        data['scaler'] = scaler

    if _DBG_DATA:
        print(f"[DBG:load_data] load_dataset done  dataset={dataset_name}  "
              f"train_batches={len(data['train_loader'])}  "
              f"val_batches={len(data['val_loader'])}  "
              f"test_batches={len(data['test_loader'])}")
    return data


# ─────────── Adjacent matrix loading & preprocessing ─────────

def load_adj(file_path, adj_type):
    """
    Load raw adjacency and return the preprocessed version(s)
    according to *adj_type*.

    Returns (processed_adj_list, raw_adj).
    """
    try:
        sensor_ids, sensor_id_to_ind, raw_adj = load_pickle(file_path)
    except (ValueError, TypeError):
        raw_adj = load_pickle(file_path)

    if _DBG_DATA:
        shape_info = raw_adj.shape if hasattr(raw_adj, 'shape') else "unknown"
        print(f"[DBG:load_data] load_adj  path={file_path}  adj_type={adj_type}  shape={shape_info}")

    _adj_builders = {
        "scalap":          lambda a: [calculate_scaled_laplacian(a).astype(np.float32).todense()],
        "normlap":         lambda a: [calculate_symmetric_normalized_laplacian(a).astype(np.float32).todense()],
        "symnadj":         lambda a: [symmetric_message_passing_adj(a)],
        "transition":      lambda a: [transition_matrix(a).T],
        "doubletransition": lambda a: [transition_matrix(a).T, transition_matrix(a.T).T],
        "identity":        lambda a: [np.diag(np.ones(a.shape[0])).astype(np.float32)],
        "original":        lambda a: a,
    }

    if adj_type not in _adj_builders:
        raise ValueError(f"Unknown adj_type '{adj_type}'. Choose from {list(_adj_builders.keys())}")

    processed = _adj_builders[adj_type](raw_adj)
    return processed, raw_adj
