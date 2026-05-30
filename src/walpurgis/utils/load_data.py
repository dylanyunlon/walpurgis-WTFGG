#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Data loading utilities for Walpurgis engine.

Walpurgis adaptations vs upstream D2STGNN:
- Every load/transform step prints shape, dtype, and memory footprint
- Memory estimates for loaded data arrays with tier placement hints
- Normalization parameters are logged for reproducibility
- Data integrity checks (NaN, Inf, range, outlier) on loaded arrays
- Timing breakdown for each I/O and transform phase
- Tier budget estimation: reports whether data fits in HBM/GDDR/DRAM
- dump_dataset_summary() for checkpoint-time diagnostics

Debug pattern — at any point during training, call:
    from utils.load_data import dump_dataset_summary
    dump_dataset_summary(data_dict, dataset_name)
"""
import pickle
import os
import time
import sys

import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *

# ── Walpurgis: global load history for retrospective debugging ──────────
_load_history = []


def dump_dataset_summary(data_dict, dataset_name="unknown"):
    """Print a comprehensive summary of the loaded dataset.
    
    Call this at any point (e.g., after epoch 0) to verify data integrity.
    Useful for debugging: if training diverges, check that data looks sane.
    """
    print(f"\n{'='*65}")
    print(f"[Walpurgis] Dataset Summary: {dataset_name}")
    print(f"{'='*65}")
    for split in ['train', 'val', 'test']:
        x_key, y_key = f'x_{split}', f'y_{split}'
        if x_key in data_dict and y_key in data_dict:
            x, y = data_dict[x_key], data_dict[y_key]
            print(f"  {split}: x={x.shape} y={y.shape} "
                  f"x_range=[{x.min():.4f},{x.max():.4f}] "
                  f"y_range=[{y.min():.4f},{y.max():.4f}] "
                  f"x_mem={x.nbytes/1e6:.1f}MB y_mem={y.nbytes/1e6:.1f}MB")
            # NaN/Inf check
            x_nan = np.isnan(x).sum()
            y_nan = np.isnan(y).sum()
            if x_nan > 0 or y_nan > 0:
                print(f"    ⚠ NaN: x={x_nan} y={y_nan}")
    if 'scaler' in data_dict:
        scaler = data_dict['scaler']
        if hasattr(scaler, 'mean'):
            print(f"  scaler: StandardScaler(mean={scaler.mean:.6f}, std={scaler.std:.6f})")
        elif callable(scaler):
            print(f"  scaler: {scaler.__name__} (MinMax)")
    print(f"  load_history: {len(_load_history)} files loaded")
    print(f"{'='*65}\n")


def re_normalization(x, mean, std):
    """Standard re-normalization: x * std + mean."""
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    """Max-min normalization to [-1, 1] range.
    
    Formula: x_norm = 2 * (x - min) / (max - min) - 1
    """
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
    statistics on first few uses for debugging normalization issues.
    
    Debug pattern:
        # During training, print scaler state:
        print(f"scaler stats: {data_dict['scaler'].get_stats()}")
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._transform_count = 0
        self._inv_count = 0
        self._transform_ranges = []  # Track output ranges for anomaly detection
        print(f"[Walpurgis::StandardScaler] init mean={mean:.6f} std={std:.6f}")
        if std < 1e-8:
            print(f"  ⚠ WARNING: std is near zero ({std:.2e}) — division instability likely!")

    def transform(self, data):
        self._transform_count += 1
        result = (data - self.mean) / self.std
        if self._transform_count <= 3:
            if isinstance(data, np.ndarray):
                out_range = (result.min(), result.max())
                self._transform_ranges.append(out_range)
                print(f"[Walpurgis::StandardScaler::transform] call#{self._transform_count} "
                      f"shape={data.shape} "
                      f"input range=[{data.min():.4f},{data.max():.4f}] "
                      f"output range=[{out_range[0]:.4f},{out_range[1]:.4f}]")
        return result

    def inverse_transform(self, data):
        self._inv_count += 1
        result = (data * self.std) + self.mean
        if self._inv_count <= 3:
            print(f"[Walpurgis::StandardScaler::inverse] call#{self._inv_count} "
                  f"output range=[{np.min(result):.4f},{np.max(result):.4f}]")
        return result

    def get_stats(self):
        """Return scaler usage statistics for debugging."""
        return {
            'mean': self.mean, 'std': self.std,
            'transform_calls': self._transform_count,
            'inverse_calls': self._inv_count,
            'observed_ranges': self._transform_ranges[-5:],
        }


def load_pickle(pickle_file):
    """Load pickle data with error handling, size reporting, and timing."""
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
    _load_history.append({'file': pickle_file, 'size_kb': file_size, 'time_s': elapsed})
    print(f"[Walpurgis::load_pickle] loaded {pickle_file} "
          f"({file_size:.1f} KB) in {elapsed:.3f}s")
    return pickle_data


def _check_array_health(name, arr):
    """Check numpy array for NaN, Inf, and suspicious ranges.
    
    Also estimates tier placement based on size:
    - < 1 MB: fits in any tier
    - 1-10 MB: should be in HBM or GDDR
    - > 10 MB: HBM preferred, may need migration strategy
    """
    if not isinstance(arr, np.ndarray):
        return
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    mem_mb = arr.nbytes / (1024 * 1024)
    
    # Tier hint
    if mem_mb < 1:
        tier_hint = "any-tier"
    elif mem_mb < 10:
        tier_hint = "HBM/GDDR"
    else:
        tier_hint = "HBM-preferred"
    
    flag = ""
    if has_nan:
        flag += " ⚠NaN"
    if has_inf:
        flag += " ⚠Inf"
    
    # Outlier detection: check for values > 5 std from mean
    mean_val = arr.mean()
    std_val = arr.std()
    if std_val > 0:
        outlier_count = np.sum(np.abs(arr - mean_val) > 5 * std_val)
        if outlier_count > 0:
            flag += f" ⚠{outlier_count}_outliers(>5σ)"
    
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
          f"range=[{arr.min():.4f},{arr.max():.4f}] "
          f"mean={mean_val:.4f} std={std_val:.4f} "
          f"mem={mem_mb:.2f}MB [{tier_hint}]{flag}")


def _estimate_tier_budget(data_dict):
    """Estimate total memory footprint and suggest tier placement."""
    total_bytes = 0
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            total_bytes += val.nbytes
    total_mb = total_bytes / (1024 * 1024)
    
    if total_mb < 50:
        suggestion = "All data fits in HBM — single-tier sufficient"
    elif total_mb < 200:
        suggestion = "Data fits in HBM+GDDR — two-tier recommended"
    else:
        suggestion = "Data exceeds GDDR — three-tier (HBM+GDDR+DRAM) needed"
    
    print(f"\n[Walpurgis::tier_budget] total_data={total_mb:.1f}MB → {suggestion}")
    return total_mb


def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """Load the complete dataset with Walpurgis diagnostics.

    Returns data_dict with train/val/test loaders and scaler.
    
    Debug: after loading, call dump_dataset_summary(data_dict, dataset_name)
    to get a full printout of all array shapes, ranges, and health.
    """
    print(f"\n{'='*65}")
    print(f"[Walpurgis::load_dataset] Loading {dataset_name} from {data_dir}")
    print(f"  batch_size: train={batch_size} val={valid_batch_size} test={test_batch_size}")
    print(f"  python={sys.version.split()[0]} numpy={np.__version__}")
    print(f"{'='*65}")

    t0_total = time.perf_counter()
    data_dict = {}

    # Phase 1: Read raw data
    print("\n── Phase 1: Raw data loading ──")
    for mode in ['train', 'val', 'test']:
        t0 = time.perf_counter()
        fpath = os.path.join(data_dir, mode + '.npz')
        _ = np.load(fpath)
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']
        elapsed = time.perf_counter() - t0
        print(f"  [{mode}] loaded {fpath} in {elapsed:.3f}s")
        _check_array_health(f"x_{mode}", data_dict['x_' + mode])
        _check_array_health(f"y_{mode}", data_dict['y_' + mode])

    # Phase 2: Normalization
    print("\n── Phase 2: Normalization ──")
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':    # traffic flow
        print(f"[Walpurgis] Traffic flow dataset — using MinMax normalization")
        _min = pickle.load(open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(open("datasets/" + dataset_name + "/max.pkl", 'rb'))
        
        _min_arr = np.array(_min)
        _max_arr = np.array(_max)
        print(f"  _min shape={_min_arr.shape} range=[{_min_arr.min():.4f},{_min_arr.max():.4f}]")
        print(f"  _max shape={_max_arr.shape} range=[{_max_arr.min():.4f},{_max_arr.max():.4f}]")
        
        # Check for degenerate normalization (max ≈ min)
        range_vals = _max_arr - _min_arr
        zero_range = np.sum(np.abs(range_vals) < 1e-8)
        if zero_range > 0:
            print(f"  ⚠ {zero_range} features have zero range (max≈min) — normalization unstable")

        # Apply normalization
        t0_norm = time.perf_counter()
        y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

        y_train_new = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train'] = np.transpose(y_train_new, axes=[0, 2, 1])
        y_val_new = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val'] = np.transpose(y_val_new, axes=[0, 2, 1])
        y_test_new = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test'] = np.transpose(y_test_new, axes=[0, 2, 1])
        
        norm_elapsed = time.perf_counter() - t0_norm
        print(f"  MinMax normalization applied in {norm_elapsed:.3f}s")
        print(f"  y_train normalized: range=[{data_dict['y_train'].min():.4f},{data_dict['y_train'].max():.4f}]")

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = re_max_min_normalization

    else:   # traffic speed
        print(f"[Walpurgis] Traffic speed dataset — using Standard normalization")
        raw_mean = data_dict['x_train'][..., 0].mean()
        raw_std = data_dict['x_train'][..., 0].std()
        print(f"  computed from x_train[...,0]: mean={raw_mean:.6f} std={raw_std:.6f}")
        
        scaler = StandardScaler(mean=raw_mean, std=raw_std)

        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = scaler

    # Phase 3: Summary
    total_elapsed = time.perf_counter() - t0_total
    total_samples = sum(len(data_dict[f'x_{m}']) for m in ['train', 'val', 'test'])
    total_mem = sum(data_dict[f'x_{m}'].nbytes + data_dict[f'y_{m}'].nbytes for m in ['train', 'val', 'test'])
    
    print(f"\n── Phase 3: Summary ──")
    print(f"[Walpurgis::load_dataset] COMPLETE in {total_elapsed:.3f}s")
    print(f"  total_samples={total_samples} total_mem={total_mem / (1024**2):.1f}MB")
    print(f"  loader batches: train={len(data_dict['train_loader'])} "
          f"val={len(data_dict['val_loader'])} test={len(data_dict['test_loader'])}")
    print(f"  samples/split: train={len(data_dict['x_train'])} "
          f"val={len(data_dict['x_val'])} test={len(data_dict['x_test'])}")
    
    _estimate_tier_budget(data_dict)

    return data_dict


def load_adj(file_path, adj_type):
    """Load adjacency matrix and preprocess it.

    Walpurgis: reports matrix properties (shape, density, symmetry, connectivity).
    Timing breakdown for pickle load + matrix computation.
    """
    print(f"\n[Walpurgis::load_adj] file={file_path} type={adj_type}")
    t0 = time.perf_counter()

    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
        print(f"  sensor_ids: {len(sensor_ids)} sensors")
        print(f"  id mapping: {list(sensor_id_to_ind.items())[:3]}... (first 3)")
    except:
        # PEMS04 / PEMS08
        adj_mx = load_pickle(file_path)

    # Preprocessing based on adj_type
    t0_proc = time.perf_counter()
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
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        raise ValueError(f"[Walpurgis] Unknown adj_type: '{adj_type}'. "
                        f"Valid: scalap, normlap, symnadj, transition, doubletransition, identity, original")
    proc_elapsed = time.perf_counter() - t0_proc

    total_elapsed = time.perf_counter() - t0

    # Walpurgis: adjacency diagnostics
    if isinstance(adj_mx, np.ndarray):
        n_nodes = adj_mx.shape[0]
        nnz = np.count_nonzero(adj_mx)
        density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
        is_symmetric = np.allclose(adj_mx, adj_mx.T)
        has_self_loops = np.any(np.diag(adj_mx) != 0)
        max_degree = np.array(adj_mx > 0).sum(axis=1).max()
        min_degree = np.array(adj_mx > 0).sum(axis=1).min()
        print(f"  raw adj: {n_nodes}×{n_nodes} nnz={nnz} density={density:.4f} "
              f"symmetric={is_symmetric} self_loops={has_self_loops}")
        print(f"  degree range: [{min_degree}, {max_degree}]")

    if isinstance(adj, list):
        for i, a in enumerate(adj):
            a_arr = np.array(a)
            adj_mem_kb = a_arr.nbytes / 1024
            print(f"  processed adj[{i}]: shape={a_arr.shape} "
                  f"range=[{a_arr.min():.4f},{a_arr.max():.4f}] "
                  f"mem={adj_mem_kb:.1f}KB")

    print(f"[Walpurgis::load_adj] done in {total_elapsed:.3f}s "
          f"(proc={proc_elapsed:.3f}s)")
    return adj, adj_mx
