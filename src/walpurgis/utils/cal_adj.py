#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Adjacency matrix calculation utilities for Walpurgis engine.

Provides various graph normalization and Laplacian computation methods
used by both static and dynamic graph components.

Walpurgis adaptations:
- check_nan_inf now returns detailed diagnostics (count, positions)
- remove_nan_inf tracks how many values were sanitized
- All matrix operations print shapes and key statistics on first call
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

# Walpurgis: track sanitization calls globally
_sanitize_stats = {'nan_removed': 0, 'inf_removed': 0, 'calls': 0}


def get_sanitize_stats():
    """Return global NaN/Inf sanitization statistics."""
    return _sanitize_stats.copy()


def check_nan_inf(tensor, raise_ex=True):
    """Check tensor for NaN and Inf values.

    Walpurgis: enhanced with count reporting.
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan = torch.any(nan_mask)
    inf = torch.any(inf_mask)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()
    total = tensor.numel()

    if (nan or inf):
        print(f"[Walpurgis::check_nan_inf] ⚠ nan={nan_count}/{total} "
              f"inf={inf_count}/{total} "
              f"({(nan_count + inf_count) / total * 100:.4f}% contaminated)")
        if raise_ex:
            raise Exception({"nan": nan, "inf": inf,
                            "nan_count": nan_count, "inf_count": inf_count})
    return {"nan": nan, "inf": inf, "nan_count": nan_count, "inf_count": inf_count}, nan or inf


def remove_nan_inf(tensor):
    """Replace NaN and Inf values with zeros.

    Walpurgis: tracks cumulative sanitization statistics.
    """
    _sanitize_stats['calls'] += 1
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    _sanitize_stats['nan_removed'] += nan_count
    _sanitize_stats['inf_removed'] += inf_count

    if (nan_count > 0 or inf_count > 0) and _sanitize_stats['calls'] <= 10:
        print(f"[Walpurgis::remove_nan_inf] call#{_sanitize_stats['calls']} "
              f"removed nan={nan_count} inf={inf_count} from tensor shape={list(tensor.shape)}")

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    """Calculate Symmetric Normalized Laplacian.
    L^{Sym} = I - D^{-1/2} A D^{-1/2}
    """
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    symmetric_normalized_laplacian = (
        sp.eye(adj.shape[0]) -
        matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    )
    print(f"[Walpurgis::calc_sym_norm_lap] N={adj.shape[0]} "
          f"nnz(L)={symmetric_normalized_laplacian.nnz}")
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """Re-scaled Laplacian for Chebyshev polynomials.
    L_{scaled} = (2 / lambda_max * L) - I
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        print(f"[Walpurgis::calc_scaled_lap] computed lambda_max={lambda_max:.6f}")
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    print(f"[Walpurgis::calc_scaled_lap] N={M} lambda_max={lambda_max} nnz={L_res.nnz}")
    return L_res


def symmetric_message_passing_adj(adj):
    """Renormalized message passing adjacency (GCN-style).
    D^{-1/2} A D^{-1/2}
    """
    print("[Walpurgis::sym_mp_adj] computing renormalized message passing adj "
          "(ensure self-loop is added)")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    print(f"  result: {mp_adj.shape} range=[{mp_adj.min():.4f},{mp_adj.max():.4f}]")
    return mp_adj


def transition_matrix(adj):
    """Transition matrix P = D^{-1}A (used in DCRNN, Graph WaveNet)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    isolated = np.sum(rowsum == 0)
    if isolated > 0:
        print(f"[Walpurgis::transition_matrix] ⚠ {isolated} isolated nodes (degree=0)")
    return P
