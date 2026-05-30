#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Adjacency matrix calculation utilities for Walpurgis engine.

Provides graph normalization, Laplacian computation, and transition matrix
methods used by both static and dynamic graph components.

Walpurgis adaptations vs upstream D2STGNN:
- check_nan_inf returns detailed diagnostics (count, positions, percentage)
- remove_nan_inf tracks cumulative sanitization statistics
- All matrix operations print shapes, sparsity, and key statistics
- First-call profiling: each function logs timing on its initial invocation
- Graph topology analysis: density, symmetry, connected component hints
- dump_all_stats() for checkpoint-time summary of all matrix ops
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import time

# ── Walpurgis global tracking ──────────────────────────────────────────
_sanitize_stats = {'nan_removed': 0, 'inf_removed': 0, 'calls': 0}
_op_timing = {}  # function_name → list of elapsed times
_first_call_flags = {}  # function_name → bool (has printed first-call banner)


def get_sanitize_stats():
    """Return global NaN/Inf sanitization statistics.
    
    Useful at checkpoint time to see how many values were cleaned:
        stats = get_sanitize_stats()
        print(f"Total NaN removed: {stats['nan_removed']}")
    """
    return _sanitize_stats.copy()


def dump_all_stats():
    """Dump complete operation statistics — call at end of epoch or checkpoint.
    
    Prints:
    - Per-function call counts and total time
    - Sanitization summary
    - Average time per operation type
    """
    print("\n" + "=" * 60)
    print("[Walpurgis::cal_adj] Operation Statistics Dump")
    print("=" * 60)
    for func_name, times in _op_timing.items():
        total = sum(times)
        avg = total / len(times) if times else 0
        print(f"  {func_name}: {len(times)} calls, total={total:.4f}s, avg={avg:.4f}s")
    print(f"  sanitize: {_sanitize_stats['calls']} calls, "
          f"nan_removed={_sanitize_stats['nan_removed']}, "
          f"inf_removed={_sanitize_stats['inf_removed']}")
    print("=" * 60 + "\n")


def _record_timing(func_name, elapsed):
    """Record timing for a function call."""
    if func_name not in _op_timing:
        _op_timing[func_name] = []
    _op_timing[func_name].append(elapsed)


def _is_first_call(func_name):
    """Check if this is the first call for verbose logging."""
    if func_name not in _first_call_flags:
        _first_call_flags[func_name] = True
        return True
    return False


def check_nan_inf(tensor, raise_ex=True):
    """Check tensor for NaN and Inf values.

    Returns a diagnostics dict and a boolean flag.
    Walpurgis: enhanced with count, percentage, and position reporting.
    
    For debugging, you can set raise_ex=False and inspect the return:
        diag, has_bad = check_nan_inf(my_tensor, raise_ex=False)
        if has_bad:
            print(f"Found {diag['nan_count']} NaN at positions...")
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan = torch.any(nan_mask)
    inf = torch.any(inf_mask)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()
    total = tensor.numel()

    diagnostics = {
        "nan": nan, "inf": inf,
        "nan_count": nan_count, "inf_count": inf_count,
        "total_elements": total,
        "contamination_pct": (nan_count + inf_count) / max(total, 1) * 100,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
    }

    if nan or inf:
        print(f"[Walpurgis::check_nan_inf] ⚠ nan={nan_count}/{total} "
              f"inf={inf_count}/{total} "
              f"({diagnostics['contamination_pct']:.4f}% contaminated) "
              f"shape={list(tensor.shape)} dtype={tensor.dtype}")
        if raise_ex:
            raise Exception(diagnostics)

    return diagnostics, nan or inf


def remove_nan_inf(tensor):
    """Replace NaN and Inf values with zeros.

    Walpurgis: tracks cumulative sanitization statistics.
    First 10 calls print detailed reports for debugging.
    """
    _sanitize_stats['calls'] += 1
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    _sanitize_stats['nan_removed'] += nan_count
    _sanitize_stats['inf_removed'] += inf_count

    if (nan_count > 0 or inf_count > 0) and _sanitize_stats['calls'] <= 10:
        print(f"[Walpurgis::remove_nan_inf] call#{_sanitize_stats['calls']} "
              f"removed nan={nan_count} inf={inf_count} "
              f"from tensor shape={list(tensor.shape)} dtype={tensor.dtype}")

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    """Calculate Symmetric Normalized Laplacian.
    
    L^{Sym} = I - D^{-1/2} A D^{-1/2}
    
    For node i,j where i≠j: L^{sym}_{ij} ≤ 0.
    Eigenvalues are in [0, 2] for connected graphs.
    """
    t0 = time.perf_counter()
    first = _is_first_call('sym_norm_lap')
    
    adj = sp.coo_matrix(adj)
    n_nodes = adj.shape[0]
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    symmetric_normalized_laplacian = (
        sp.eye(n_nodes) -
        matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    )
    
    elapsed = time.perf_counter() - t0
    _record_timing('sym_norm_lap', elapsed)
    
    # Diagnostics
    nnz = symmetric_normalized_laplacian.nnz
    density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
    zero_degree = np.sum(D.flatten() == 0)
    
    print(f"[Walpurgis::calc_sym_norm_lap] N={n_nodes} nnz(L)={nnz} "
          f"density={density:.4f} zero_degree_nodes={zero_degree} "
          f"time={elapsed:.4f}s")
    
    if first and n_nodes <= 500:
        # On first call for small graphs, print degree distribution summary
        degrees = D.flatten()
        print(f"  degree stats: min={degrees.min():.1f} max={degrees.max():.1f} "
              f"mean={degrees.mean():.1f} std={degrees.std():.1f}")
    
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """Re-scaled Laplacian for Chebyshev polynomials.
    
    L_{scaled} = (2 / lambda_max × L) - I
    
    Rescales eigenvalues to [-1, 1] for stable polynomial approximation.
    Default lambda_max=2 per GCN (Kipf & Welling, 2017).
    """
    t0 = time.perf_counter()
    
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    
    computed_lambda = False
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        computed_lambda = True
        print(f"[Walpurgis::calc_scaled_lap] computed lambda_max={lambda_max:.6f}")
    
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    
    elapsed = time.perf_counter() - t0
    _record_timing('scaled_lap', elapsed)
    
    print(f"[Walpurgis::calc_scaled_lap] N={M} lambda_max={lambda_max} "
          f"nnz={L_res.nnz} computed_lambda={computed_lambda} time={elapsed:.4f}s")
    
    return L_res


def symmetric_message_passing_adj(adj):
    """Renormalized message passing adjacency (GCN-style).
    
    Computes D^{-1/2} A D^{-1/2}.
    Assumes self-loops have already been added to adj.
    """
    t0 = time.perf_counter()
    
    print("[Walpurgis::sym_mp_adj] computing renormalized message passing adj "
          "(ensure self-loop is added)")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    
    elapsed = time.perf_counter() - t0
    _record_timing('sym_mp_adj', elapsed)
    
    # Symmetry check
    sym_err = np.abs(mp_adj - mp_adj.T).max()
    print(f"  result: {mp_adj.shape} range=[{mp_adj.min():.4f},{mp_adj.max():.4f}] "
          f"symmetry_error={sym_err:.2e} time={elapsed:.4f}s")
    
    return mp_adj


def transition_matrix(adj):
    """Transition matrix P = D^{-1}A (used in DCRNN, Graph WaveNet).
    
    Row-normalized adjacency: each row sums to 1 (or 0 for isolated nodes).
    """
    t0 = time.perf_counter()
    
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    
    elapsed = time.perf_counter() - t0
    _record_timing('transition_matrix', elapsed)
    
    # Topology diagnostics
    isolated = np.sum(rowsum == 0)
    n_nodes = adj.shape[0]
    nnz = adj.nnz
    avg_degree = rowsum.mean()
    
    if isolated > 0:
        print(f"[Walpurgis::transition_matrix] ⚠ {isolated}/{n_nodes} isolated nodes (degree=0)")
    
    row_sums_P = np.array(P.sum(axis=1)).flatten()
    non_isolated_rows = row_sums_P[rowsum > 0]
    row_sum_err = np.abs(non_isolated_rows - 1.0).max() if len(non_isolated_rows) > 0 else 0.0
    
    print(f"[Walpurgis::transition_matrix] N={n_nodes} edges={nnz} "
          f"avg_degree={avg_degree:.1f} row_sum_error={row_sum_err:.2e} "
          f"time={elapsed:.4f}s")
    
    return P
