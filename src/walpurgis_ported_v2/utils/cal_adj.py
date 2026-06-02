#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adjacency matrix computation utilities for Walpurgis spatial-temporal graph networks.
Provides Laplacian variants, transition matrices, and numerical safety helpers.
"""

import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys

# ───────────────────── Numerical Safety ─────────────────────

_DBG_ADJ = ("--debug-adj" in sys.argv) or False   # flip to True for wall-of-state prints


def check_nan_inf(tensor, label="tensor", raise_ex=True):
    """Inspect a tensor for NaN / Inf and optionally explode."""
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    summary = {"nan": has_nan.item(), "inf": has_inf.item()}
    if _DBG_ADJ:
        print(f"[DBG:cal_adj] check_nan_inf  label={label}  shape={tuple(tensor.shape)}  "
              f"nan={has_nan.item()}  inf={has_inf.item()}  "
              f"min={tensor.min().item():.6g}  max={tensor.max().item():.6g}")
    if raise_ex and (has_nan or has_inf):
        raise ValueError(f"Numerical issue in '{label}': {summary}")
    return summary, (has_nan or has_inf)


def remove_nan_inf(tensor):
    """Replace NaN and Inf entries with zeros."""
    cleaned = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    cleaned = torch.where(torch.isinf(cleaned), torch.zeros_like(cleaned), cleaned)
    if _DBG_ADJ:
        bad_count = int((torch.isnan(tensor) | torch.isinf(tensor)).sum().item())
        print(f"[DBG:cal_adj] remove_nan_inf  shape={tuple(tensor.shape)}  "
              f"bad_entries_removed={bad_count}")
    return cleaned


# ───────────────── Laplacian & Transition ───────────────────

def calculate_symmetric_normalized_laplacian(adj_matrix):
    """
    Symmetric normalized Laplacian:  L^{sym} = I - D^{-1/2} A D^{-1/2}

    For connected pairs (i,j) where i≠j, L^{sym}_{ij} ≤ 0.
    """
    sp_adj = sp.coo_matrix(adj_matrix)
    row_sum = np.array(sp_adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    diag_d_inv_sqrt = sp.diags(d_inv_sqrt)
    eye = sp.eye(sp_adj.shape[0])
    laplacian = eye - diag_d_inv_sqrt.dot(sp_adj).dot(diag_d_inv_sqrt).tocoo()
    if _DBG_ADJ:
        print(f"[DBG:cal_adj] sym_norm_laplacian  nodes={sp_adj.shape[0]}  "
              f"nnz={laplacian.nnz}  row_sum_range=[{row_sum.min():.4f}, {row_sum.max():.4f}]")
    return laplacian


def calculate_scaled_laplacian(adj_matrix, lambda_max=2, undirected=True):
    """
    Rescale eigenvalues to [-1, 1] for Chebyshev polynomial basis.
    L_scaled = (2 / λ_max) · L_sym  −  I
    """
    if undirected:
        adj_matrix = np.maximum.reduce([adj_matrix, adj_matrix.T])
    sym_lap = calculate_symmetric_normalized_laplacian(adj_matrix)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(sym_lap, 1, which='LM')
        lambda_max = lambda_max[0]
    sym_lap = sp.csr_matrix(sym_lap)
    n_nodes = sym_lap.shape[0]
    identity = sp.identity(n_nodes, format='csr', dtype=sym_lap.dtype)
    scaled = (2.0 / lambda_max * sym_lap) - identity
    if _DBG_ADJ:
        print(f"[DBG:cal_adj] scaled_laplacian  nodes={n_nodes}  lambda_max={lambda_max:.6f}  "
              f"nnz={scaled.nnz}")
    return scaled


def symmetric_message_passing_adj(adj_matrix):
    """
    GCN-style renormalized adjacency: D^{-1/2} A^T D^{-1/2}.
    Caller is responsible for ensuring self-loops exist in adj.
    """
    if _DBG_ADJ:
        diag_sum = np.trace(adj_matrix) if isinstance(adj_matrix, np.ndarray) else 0
        print(f"[DBG:cal_adj] sym_mp_adj  trace(A)={diag_sum:.4f}  "
              f"(nonzero trace → self-loops present)")
    sp_adj = sp.coo_matrix(adj_matrix)
    row_sum = np.array(sp_adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    diag_mat = sp.diags(d_inv_sqrt)
    mp_adj = diag_mat.dot(sp_adj).transpose().dot(diag_mat).astype(np.float32).todense()
    return mp_adj


def transition_matrix(adj_matrix):
    """
    Row-stochastic transition matrix: P = D^{-1} A = A / rowsum(A).
    Used in DCRNN and Graph WaveNet diffusion convolution.
    """
    sp_adj = sp.coo_matrix(adj_matrix)
    row_sum = np.array(sp_adj.sum(axis=1)).flatten()
    d_inv = np.power(row_sum, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    diag_d = sp.diags(d_inv)
    trans = diag_d.dot(sp_adj).astype(np.float32).todense()
    if _DBG_ADJ:
        print(f"[DBG:cal_adj] transition_matrix  nodes={sp_adj.shape[0]}  "
              f"row_sum_range=[{row_sum.min():.4f}, {row_sum.max():.4f}]  "
              f"P_range=[{np.min(trans):.6f}, {np.max(trans):.6f}]")
    return trans
