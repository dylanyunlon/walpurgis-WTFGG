#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Data loading utilities for Walpurgis engine.

Walpurgis adaptations:
- Every load/transform step prints shape and dtype information
- Memory estimates for loaded data arrays
- Normalization parameters are logged for reproducibility
- Data integrity checks (NaN, Inf, range) on loaded arrays
"""
import pickle
import os
import time

import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *


def re_normalization(x, mean, std):
    """Standard re-normalization: x * std + mean."""
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    """Max-min normalization to [-1, 1] range."""
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    """Inverse max-min normalization from [-1, 1] back to original range."""
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


class StandardScaler():
    """Standard scaler with Walpurgis diagnostics.

    Tracks transform/inverse_transform call counts and reports
    statistics on first use.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._transform_count = 0
        self._inv_count = 0
        print(f"[Walpurgis::StandardScaler] init mean={mean:.6f} std={std:.6f}")

    def transform(self, data):
        self._transform_count += 1
        result = (data - self.mean) / self.std
        if self._transform_count <= 3:
            if isinstance(data, np.ndarray):
                print(f"[Walpurgis::StandardScaler::transform] call#{self._transform_count} "
                      f"shape={data.shape} "
                      f"input range=[{data.min():.4f},{data.max():.4f}] "
                      f"output range=[{result.min():.4f},{result.max():.4f}]")
        return result

    def inverse_transform(self, data):
        self._inv_count += 1
        result = (data * self.std) + self.mean
        if self._inv_count <= 3:
            print(f"[Walpurgis::StandardScaler::inverse] call#{self._inv_count}")
        return result


def load_pickle(pickle_file):
    """Load pickle data with error handling and size reporting."""
    t0 = time.perf_counter()
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'[Walpurgis::load_pickle] ✗ Unable to load {pickle_file}: {e}')
        raise
    elapsed = time.perf_counter() - t0
    file_size = os.path.getsize(pickle_file) / 1024
    print(f"[Walpurgis::load_pickle] loaded {pickle_file} "
          f"({file_size:.1f} KB) in {elapsed:.3f}s")
    return pickle_data


def _check_array_health(name, arr):
    """Check numpy array for NaN, Inf, and suspicious ranges."""
    if not isinstance(arr, np.ndarray):
        return
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    if has_nan or has_inf:
        print(f"  ⚠ {name}: nan={has_nan} inf={has_inf}")
    mem_mb = arr.nbytes / (1024 * 1024)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
          f"range=[{arr.min():.4f},{arr.max():.4f}] mem={mem_mb:.2f}MB")


def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """Load the complete dataset with Walpurgis diagnostics.

    Returns data_dict with train/val/test loaders and scaler.
    """
    print(f"\n{'='*60}")
    print(f"[Walpurgis::load_dataset] Loading {dataset_name} from {data_dir}")
    print(f"  batch_size: train={batch_size} val={valid_batch_size} test={test_batch_size}")
    print(f"{'='*60}")

    t0_total = time.perf_counter()
    data_dict = {}

    # read data: train_x, train_y, val_x, val_y, test_x, test_y
    for mode in ['train', 'val', 'test']:
        t0 = time.perf_counter()
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']
        elapsed = time.perf_counter() - t0
        print(f"  [{mode}] loaded in {elapsed:.3f}s")
        _check_array_health(f"x_{mode}", data_dict['x_' + mode])
        _check_array_health(f"y_{mode}", data_dict['y_' + mode])

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':    # traffic flow
        print(f"\n[Walpurgis] Traffic flow dataset — using MinMax normalization")
        _min = pickle.load(open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(open("datasets/" + dataset_name + "/max.pkl", 'rb'))
        print(f"  _min shape={np.array(_min).shape if hasattr(_min, 'shape') else 'scalar'} "
              f"_max shape={np.array(_max).shape if hasattr(_max, 'shape') else 'scalar'}")

        # normalization
        y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

        y_train_new = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train'] = np.transpose(y_train_new, axes=[0, 2, 1])
        y_val_new = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val'] = np.transpose(y_val_new, axes=[0, 2, 1])
        y_test_new = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test'] = np.transpose(y_test_new, axes=[0, 2, 1])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = re_max_min_normalization

    else:   # traffic speed
        print(f"\n[Walpurgis] Traffic speed dataset — using Standard normalization")
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std=data_dict['x_train'][..., 0].std()
        )

        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = scaler

    total_elapsed = time.perf_counter() - t0_total
    # Summary
    total_samples = sum(len(data_dict[f'x_{m}']) for m in ['train', 'val', 'test'])
    total_mem = sum(data_dict[f'x_{m}'].nbytes + data_dict[f'y_{m}'].nbytes for m in ['train', 'val', 'test'])
    print(f"\n[Walpurgis::load_dataset] COMPLETE in {total_elapsed:.3f}s")
    print(f"  total_samples={total_samples} total_mem={total_mem / (1024**2):.1f}MB")
    print(f"  loader batches: train={len(data_dict['train_loader'])} "
          f"val={len(data_dict['val_loader'])} test={len(data_dict['test_loader'])}")

    return data_dict


def load_adj(file_path, adj_type):
    """Load adjacency matrix and preprocess it.

    Walpurgis: reports matrix properties (shape, density, symmetry).
    """
    print(f"\n[Walpurgis::load_adj] file={file_path} type={adj_type}")
    t0 = time.perf_counter()

    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
        print(f"  sensor_ids: {len(sensor_ids)} sensors")
    except:
        # PEMS04
        adj_mx = load_pickle(file_path)

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"

    elapsed = time.perf_counter() - t0

    # Walpurgis: adjacency diagnostics
    if isinstance(adj_mx, np.ndarray):
        n_nodes = adj_mx.shape[0]
        nnz = np.count_nonzero(adj_mx)
        density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
        is_symmetric = np.allclose(adj_mx, adj_mx.T)
        print(f"  adj_mx: {n_nodes}×{n_nodes} nnz={nnz} density={density:.4f} "
              f"symmetric={is_symmetric}")

    if isinstance(adj, list):
        for i, a in enumerate(adj):
            a_arr = np.array(a)
            print(f"  processed adj[{i}]: shape={a_arr.shape} "
                  f"range=[{a_arr.min():.4f},{a_arr.max():.4f}]")

    print(f"[Walpurgis::load_adj] done in {elapsed:.3f}s")
    return adj, adj_mx
