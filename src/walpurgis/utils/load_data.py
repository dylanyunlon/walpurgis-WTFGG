#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis Data Loading — Dataset Pipeline with Tier-Aware Profiling
====================================================================
Derived from D2STGNN load_data.py with ~20% restructuring.

Changes:
  1. DatasetProfile class replaces scattered globals for load history
  2. Array health checks include outlier detection and tier hints
  3. StandardScaler tracks transform ranges for anomaly detection
  4. Normalization path is selected automatically with diagnostics
  5. Tier budget estimation reports HBM/GDDR/DRAM placement
"""
import pickle
import os
import time
import sys
from collections import deque

import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *


# ═══════════ Load History Tracker ═══════════ #

class DatasetProfile:
    """Centralized dataset loading diagnostics.
    
    Call DatasetProfile.summary(data_dict, name) at any point
    to get a full printout of all loaded data.
    """
    _history = []
    
    @classmethod
    def record_load(cls, path, size_kb, elapsed):
        cls._history.append({'file': path, 'size_kb': size_kb, 'time': elapsed})
    
    @classmethod
    def summary(cls, data_dict, dataset_name="unknown"):
        """Print comprehensive dataset summary — call from debugger."""
        print(f"\n{'═'*65}")
        print(f"  Dataset Summary: {dataset_name}")
        print(f"{'═'*65}")
        for split in ['train', 'val', 'test']:
            xk, yk = f'x_{split}', f'y_{split}'
            if xk in data_dict and yk in data_dict:
                x, y = data_dict[xk], data_dict[yk]
                print(f"  {split}: x={x.shape} y={y.shape} "
                      f"x∈[{x.min():.4f},{x.max():.4f}] "
                      f"y∈[{y.min():.4f},{y.max():.4f}] "
                      f"mem={x.nbytes/1e6:.1f}+{y.nbytes/1e6:.1f}MB")
                xn = np.isnan(x).sum()
                if xn > 0:
                    print(f"    ⚠ NaN in x: {xn}")
        if 'scaler' in data_dict:
            s = data_dict['scaler']
            if hasattr(s, 'mean'):
                print(f"  scaler: StandardScaler(μ={s.mean:.6f}, σ={s.std:.6f})")
        print(f"  load_history: {len(cls._history)} files")
        print(f"{'═'*65}\n")


# ═══════════ Normalization Utilities ═══════════ #

def re_normalization(x, mean, std):
    """Inverse standard normalization: x·σ + μ."""
    return x * std + mean


def max_min_normalization(x, _max, _min):
    """Scale to [-1, 1]: 2·(x - min)/(max - min) - 1."""
    return 2.0 * (x - _min) / (_max - _min) - 1.0


def re_max_min_normalization(x, _max, _min):
    """Inverse of max_min_normalization."""
    return 0.5 * (x + 1.0) * (_max - _min) + _min


class StandardScaler:
    """Standard normalization with diagnostic tracking.
    
    Monitors transform output ranges for anomaly detection.
    Call scaler.stats() to inspect from debugger.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._n_transform = 0
        self._n_inverse = 0
        self._ranges = deque(maxlen=10)
        print(f"[StandardScaler] μ={mean:.6f} σ={std:.6f}")
        if std < 1e-8:
            print(f"  ⚠ σ near zero ({std:.2e}) — division instability")

    def transform(self, data):
        self._n_transform += 1
        result = (data - self.mean) / self.std
        if self._n_transform <= 3 and isinstance(data, np.ndarray):
            rng = (float(result.min()), float(result.max()))
            self._ranges.append(rng)
            print(f"[Scaler::transform #{self._n_transform}] "
                  f"{data.shape} in=[{data.min():.4f},{data.max():.4f}] "
                  f"out=[{rng[0]:.4f},{rng[1]:.4f}]")
        return result

    def inverse_transform(self, data):
        self._n_inverse += 1
        result = data * self.std + self.mean
        if self._n_inverse <= 3:
            print(f"[Scaler::inverse #{self._n_inverse}] "
                  f"out∈[{np.min(result):.4f},{np.max(result):.4f}]")
        return result

    def stats(self):
        """Return scaler state — call from debugger."""
        return {'mean': self.mean, 'std': self.std,
                'transforms': self._n_transform,
                'inverses': self._n_inverse,
                'recent_ranges': list(self._ranges)}


# ═══════════ File I/O ═══════════ #

def load_pickle(pickle_file):
    """Load pickle with timing, error handling, and size reporting."""
    t0 = time.perf_counter()
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'[load_pickle] ✗ {pickle_file}: {e}')
        raise
    elapsed = time.perf_counter() - t0
    size_kb = os.path.getsize(pickle_file) / 1024
    DatasetProfile.record_load(pickle_file, size_kb, elapsed)
    print(f"[load_pickle] {pickle_file} ({size_kb:.1f}KB) in {elapsed:.3f}s")
    return data


def _inspect_array(label, arr):
    """Array health check: NaN, Inf, outliers, tier hint."""
    if not isinstance(arr, np.ndarray):
        return
    mem_mb = arr.nbytes / (1024 * 1024)
    tier = "any" if mem_mb < 1 else ("HBM/GDDR" if mem_mb < 10 else "HBM")
    
    flags = ""
    if np.isnan(arr).any(): flags += " ⚠NaN"
    if np.isinf(arr).any(): flags += " ⚠Inf"
    
    mu, sigma = arr.mean(), arr.std()
    if sigma > 0:
        n_outlier = int(np.sum(np.abs(arr - mu) > 5 * sigma))
        if n_outlier > 0:
            flags += f" ⚠{n_outlier}_outliers"
    
    print(f"  {label}: {arr.shape} {arr.dtype} "
          f"∈[{arr.min():.4f},{arr.max():.4f}] "
          f"μ={mu:.4f} σ={sigma:.4f} "
          f"{mem_mb:.2f}MB [{tier}]{flags}")


def _tier_budget(data_dict):
    """Estimate total memory and suggest tier placement."""
    total = sum(v.nbytes for v in data_dict.values() if isinstance(v, np.ndarray))
    mb = total / (1024 * 1024)
    if mb < 50:
        hint = "single-tier (HBM) sufficient"
    elif mb < 200:
        hint = "two-tier (HBM+GDDR) recommended"
    else:
        hint = "three-tier (HBM+GDDR+DRAM) needed"
    print(f"\n[tier_budget] {mb:.1f}MB total → {hint}")
    return mb


# ═══════════ Main Data Loading ═══════════ #

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """Load dataset with full diagnostics and tier budget estimation.
    
    After loading, call DatasetProfile.summary(result, dataset_name) for details.
    """
    print(f"\n{'═'*65}")
    print(f"  Loading {dataset_name} from {data_dir}")
    print(f"  batch: train={batch_size} val={valid_batch_size} test={test_batch_size}")
    print(f"{'═'*65}")

    t_total = time.perf_counter()
    data = {}

    # Phase 1: Raw data
    print("\n── Raw data ──")
    for split in ['train', 'val', 'test']:
        t0 = time.perf_counter()
        fpath = os.path.join(data_dir, split + '.npz')
        npz = np.load(fpath)
        data[f'x_{split}'] = npz['x']
        data[f'y_{split}'] = npz['y']
        print(f"  [{split}] {fpath} in {time.perf_counter()-t0:.3f}s")
        _inspect_array(f"x_{split}", data[f'x_{split}'])
        _inspect_array(f"y_{split}", data[f'y_{split}'])

    # Phase 2: Normalization (auto-detect by dataset)
    print("\n── Normalization ──")
    is_flow = dataset_name in ('PEMS04', 'PEMS08')
    
    if is_flow:
        print(f"  Traffic flow → MinMax normalization")
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))
        
        _min_a, _max_a = np.array(_min), np.array(_max)
        range_vals = _max_a - _min_a
        n_degenerate = int(np.sum(np.abs(range_vals) < 1e-8))
        if n_degenerate > 0:
            print(f"  ⚠ {n_degenerate} features have zero range")
        
        t0 = time.perf_counter()
        for split in ['train', 'val', 'test']:
            y = np.squeeze(np.transpose(data[f'y_{split}'], axes=[0, 2, 1, 3]), axis=-1)
            y_norm = max_min_normalization(y, _max[:, :, 0, :], _min[:, :, 0, :])
            data[f'y_{split}'] = np.transpose(y_norm, axes=[0, 2, 1])
        print(f"  MinMax applied in {time.perf_counter()-t0:.3f}s")
        print(f"  y_train normalized: ∈[{data['y_train'].min():.4f},{data['y_train'].max():.4f}]")

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader']   = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader']  = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = re_max_min_normalization
    else:
        print(f"  Traffic speed → Standard normalization")
        raw_mean = float(data['x_train'][..., 0].mean())
        raw_std = float(data['x_train'][..., 0].std())
        print(f"  computed μ={raw_mean:.6f} σ={raw_std:.6f}")

        scaler = StandardScaler(mean=raw_mean, std=raw_std)
        for split in ['train', 'val', 'test']:
            data[f'x_{split}'][..., 0] = scaler.transform(data[f'x_{split}'][..., 0])
            data[f'y_{split}'][..., 0] = scaler.transform(data[f'y_{split}'][..., 0])

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader']   = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader']  = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scaler

    # Phase 3: Summary
    elapsed = time.perf_counter() - t_total
    n_samples = sum(len(data[f'x_{s}']) for s in ['train', 'val', 'test'])
    print(f"\n── Summary ──")
    print(f"  {elapsed:.3f}s total | {n_samples} samples | "
          f"loaders: {len(data['train_loader'])}/{len(data['val_loader'])}/{len(data['test_loader'])}")
    _tier_budget(data)

    return data


def load_adj(file_path, adj_type):
    """Load and preprocess adjacency matrix with topology diagnostics."""
    print(f"\n[load_adj] file={file_path} type={adj_type}")
    t0 = time.perf_counter()

    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
        print(f"  {len(sensor_ids)} sensors")
    except:
        adj_mx = load_pickle(file_path)

    # Compute requested adjacency form
    t_proc = time.perf_counter()
    type_map = {
        'scalap':          lambda a: [calculate_scaled_laplacian(a).astype(np.float32).todense()],
        'normlap':         lambda a: [calculate_symmetric_normalized_laplacian(a).astype(np.float32).todense()],
        'symnadj':         lambda a: [symmetric_message_passing_adj(a).astype(np.float32).todense()],
        'transition':      lambda a: [transition_matrix(a).T],
        'doubletransition': lambda a: [transition_matrix(a).T, transition_matrix(a.T).T],
        'identity':        lambda a: [np.diag(np.ones(a.shape[0])).astype(np.float32)],
        'original':        lambda a: a,
    }
    if adj_type not in type_map:
        raise ValueError(f"Unknown adj_type '{adj_type}'. "
                         f"Valid: {list(type_map.keys())}")
    adj = type_map[adj_type](adj_mx)
    proc_ms = (time.perf_counter() - t_proc) * 1000

    # Topology diagnostics
    if isinstance(adj_mx, np.ndarray):
        N = adj_mx.shape[0]
        nnz = np.count_nonzero(adj_mx)
        density = nnz / (N * N) if N > 0 else 0
        sym = bool(np.allclose(adj_mx, adj_mx.T))
        deg = np.array(adj_mx > 0).sum(axis=1)
        print(f"  raw: {N}×{N} nnz={nnz} density={density:.4f} "
              f"sym={sym} deg∈[{deg.min()},{deg.max()}]")

    if isinstance(adj, list):
        for i, a in enumerate(adj):
            a_np = np.array(a)
            print(f"  adj[{i}]: {a_np.shape} ∈[{a_np.min():.4f},{a_np.max():.4f}] "
                  f"{a_np.nbytes/1024:.1f}KB")

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[load_adj] done in {total_ms:.1f}ms (proc={proc_ms:.1f}ms)")
    return adj, adj_mx
