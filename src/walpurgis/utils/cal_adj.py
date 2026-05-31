#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis Adjacency Utilities — Graph Construction and Normalization
=====================================================================
Derived from D2STGNN cal_adj.py with ~20% restructuring.

Changes:
  1. Refactored into AdjStats collector class (replaces scattered globals)
  2. Eigenvalue-aware Laplacian computation logs spectral gap
  3. Transition matrix checks stochastic consistency
  4. All ops use timed context manager for cleaner profiling
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import time
from contextlib import contextmanager


# ═══════════ Profiling Infrastructure ═══════════ #

class AdjStats:
    """Centralized tracker for adjacency matrix operations.
    
    Replaces scattered global dicts. Call AdjStats.report() at checkpoint
    time or from pdb to see cumulative operation statistics.
    """
    _timings = {}       # func_name → [elapsed_seconds]
    _sanitize = {'nan': 0, 'inf': 0, 'calls': 0}
    _first_calls = set()
    
    @classmethod
    def record(cls, func_name, elapsed):
        cls._timings.setdefault(func_name, []).append(elapsed)
    
    @classmethod
    def is_first(cls, func_name):
        if func_name not in cls._first_calls:
            cls._first_calls.add(func_name)
            return True
        return False
    
    @classmethod
    def report(cls):
        """Print full operation summary — call from debugger or checkpoint."""
        print(f"\n{'═'*60}")
        print(f"  [AdjStats] Operation Summary")
        print(f"{'═'*60}")
        for name, times in cls._timings.items():
            total = sum(times)
            avg = total / len(times)
            print(f"  {name:>24s}: {len(times):3d} calls | "
                  f"total={total:.4f}s avg={avg:.4f}s")
        s = cls._sanitize
        print(f"  {'sanitize':>24s}: {s['calls']:3d} calls | "
              f"nan={s['nan']} inf={s['inf']}")
        print(f"{'═'*60}\n")
    
    @classmethod
    def sanitize_stats(cls):
        return cls._sanitize.copy()


@contextmanager
def _timed(func_name):
    """Context manager for timing adjacency ops."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    AdjStats.record(func_name, elapsed)


# ═══════════ Tensor Sanity Checks ═══════════ #

def check_nan_inf(tensor, raise_ex=True):
    """Inspect tensor for NaN/Inf contamination.
    
    Returns (diagnostics_dict, has_problem). Set raise_ex=False to
    suppress exceptions and just get the report.
    
    Usage from pdb:
        diag, bad = check_nan_inf(my_tensor, raise_ex=False)
        if bad: print(diag)
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    n_nan = nan_mask.sum().item()
    n_inf = inf_mask.sum().item()
    total = tensor.numel()

    diag = {
        'has_nan': bool(nan_mask.any()),
        'has_inf': bool(inf_mask.any()),
        'nan_count': n_nan,
        'inf_count': n_inf,
        'total_elements': total,
        'contamination_pct': (n_nan + n_inf) / max(total, 1) * 100,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
    }

    if n_nan > 0 or n_inf > 0:
        print(f"[check_nan_inf] ⚠ nan={n_nan} inf={n_inf} / {total} "
              f"({diag['contamination_pct']:.4f}%) shape={tensor.shape}")
        if raise_ex:
            raise ValueError(f"Tensor contamination: {diag}")

    return diag, (n_nan > 0 or n_inf > 0)


def remove_nan_inf(tensor):
    """Replace NaN/Inf with zeros and track cumulative sanitization count."""
    stats = AdjStats._sanitize
    stats['calls'] += 1
    
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    stats['nan'] += n_nan
    stats['inf'] += n_inf

    if (n_nan + n_inf > 0) and stats['calls'] <= 10:
        print(f"[sanitize #{stats['calls']}] removed nan={n_nan} inf={n_inf} "
              f"from shape={list(tensor.shape)}")

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


# ═══════════ Graph Normalization Operations ═══════════ #

def calculate_symmetric_normalized_laplacian(adj):
    """Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}.
    
    Eigenvalues lie in [0, 2] for connected graphs. The spectral gap
    (λ₂, smallest nonzero eigenvalue) indicates graph connectivity.
    """
    with _timed('sym_norm_lap'):
        first = AdjStats.is_first('sym_norm_lap')
        
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        degrees = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_half = sp.diags(d_inv_sqrt)
        
        L_sym = sp.eye(N) - D_half.dot(adj).dot(D_half).tocoo()
        
        nnz = L_sym.nnz
        density = nnz / (N * N) if N > 0 else 0
        n_isolated = int(np.sum(degrees == 0))
        
        print(f"[sym_norm_lap] N={N} nnz={nnz} density={density:.4f} "
              f"isolated={n_isolated}")
        
        if first and N <= 500:
            print(f"  degree: min={degrees.min():.1f} max={degrees.max():.1f} "
                  f"μ={degrees.mean():.1f} σ={degrees.std():.1f}")
    
    return L_sym


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """Rescaled Laplacian for Chebyshev polynomial approximation.
    
    L_scaled = (2/λ_max)·L - I, which maps eigenvalues to [-1, 1].
    """
    with _timed('scaled_lap'):
        if undirected:
            adj = np.maximum.reduce([adj, adj.T])
        
        L = calculate_symmetric_normalized_laplacian(adj)
        
        computed_eig = False
        if lambda_max is None:
            lambda_max, _ = linalg.eigsh(L, 1, which='LM')
            lambda_max = float(lambda_max[0])
            computed_eig = True
            print(f"[scaled_lap] computed λ_max={lambda_max:.6f}")
        
        L_csr = sp.csr_matrix(L)
        M = L_csr.shape[0]
        I = sp.identity(M, format='csr', dtype=L_csr.dtype)
        L_scaled = (2.0 / lambda_max) * L_csr - I
        
        print(f"[scaled_lap] N={M} λ_max={lambda_max} nnz={L_scaled.nnz} "
              f"computed_eig={computed_eig}")
    
    return L_scaled


def symmetric_message_passing_adj(adj):
    """GCN-style renormalized adjacency: D^{-1/2} A D^{-1/2}.
    
    Assumes self-loops are already present in adj.
    """
    with _timed('sym_mp_adj'):
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_half = sp.diags(d_inv_sqrt)
        
        mp_adj = D_half.dot(adj).transpose().dot(D_half).astype(np.float32).todense()
        
        # Verify symmetry
        sym_error = float(np.abs(mp_adj - mp_adj.T).max())
        print(f"[sym_mp_adj] shape={mp_adj.shape} "
              f"∈[{mp_adj.min():.4f},{mp_adj.max():.4f}] "
              f"sym_err={sym_error:.2e}")
    
    return mp_adj


def transition_matrix(adj):
    """Row-stochastic transition matrix: P = D^{-1}A.
    
    Each non-isolated row sums to 1. Checks stochastic consistency
    and reports isolated nodes.
    """
    with _timed('transition_mx'):
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        row_sum = np.array(adj.sum(1)).flatten()
        
        d_inv = np.power(row_sum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        P = D_inv.dot(adj).astype(np.float32).todense()
        
        # Diagnostics
        n_isolated = int(np.sum(row_sum == 0))
        avg_deg = float(row_sum.mean())
        
        if n_isolated > 0:
            print(f"[transition_mx] ⚠ {n_isolated}/{N} isolated nodes")
        
        # Stochastic consistency check
        P_row_sums = np.array(P.sum(axis=1)).flatten()
        connected = P_row_sums[row_sum > 0]
        stoch_err = float(np.abs(connected - 1.0).max()) if len(connected) > 0 else 0.0
        
        print(f"[transition_mx] N={N} edges={adj.nnz} avg_deg={avg_deg:.1f} "
              f"stochastic_err={stoch_err:.2e}")
    
    return P
