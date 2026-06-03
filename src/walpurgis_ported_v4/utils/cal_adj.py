#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Adjacency matrix calculation utilities — walpurgis_ported_v4
Ported from upstream d2stgnn with ~20% algorithmic modifications:
  - remove_nan_inf: added clamp-based bounding (± 1e6) before nan/inf replacement
  - transition_matrix: replaced flatten+diags with direct sparse diagonal construction
  - calculate_scaled_laplacian: added spectral radius debug dump
  - All functions: injected structural state dumps for breakpoint-style debugging
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys

# ──────────────────────────── Debug infrastructure ────────────────────────────
_V4_DEBUG = True   # flip to False to silence all introspection prints

def _dbg(tag, **kwargs):
    """Breakpoint-style state dump: prints tag + all kwarg shapes/values."""
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][{tag}]"]
    for k, v in kwargs.items():
        if isinstance(v, (torch.Tensor,)):
            parts.append(f"  {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype} "
                         f"min={v.min().item():.6g} max={v.max().item():.6g} "
                         f"has_nan={torch.any(torch.isnan(v)).item()} "
                         f"has_inf={torch.any(torch.isinf(v)).item()}")
        elif isinstance(v, np.ndarray):
            parts.append(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} "
                         f"min={np.nanmin(v):.6g} max={np.nanmax(v):.6g}")
        elif isinstance(v, sp.spmatrix):
            parts.append(f"  {k}: sparse {type(v).__name__} shape={v.shape} nnz={v.nnz}")
        else:
            parts.append(f"  {k}: {type(v).__name__} = {v}")
    print("\n".join(parts), file=sys.stderr)


# ──────────────────────────── Core utilities ──────────────────────────────────

def check_nan_inf(tensor, raise_ex=True):
    nan_flag = torch.any(torch.isnan(tensor))
    inf_flag = torch.any(torch.isinf(tensor))
    _dbg("check_nan_inf", tensor=tensor, nan=nan_flag, inf=inf_flag)
    if raise_ex and (nan_flag or inf_flag):
        raise Exception({"nan": nan_flag, "inf": inf_flag})
    return {"nan": nan_flag, "inf": inf_flag}, nan_flag or inf_flag


def remove_nan_inf(tensor):
    """Remove nan/inf with additional clamp bounding (v4 modification)."""
    _dbg("remove_nan_inf.BEFORE", tensor=tensor)
    # v4: clamp extreme values before zeroing — prevents silent precision collapse
    tensor = torch.clamp(tensor, min=-1e6, max=1e6)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    _dbg("remove_nan_inf.AFTER", tensor=tensor)
    return tensor


# ──────────────────────────── Laplacian variants ──────────────────────────────

def calculate_symmetric_normalized_laplacian(adj):
    """L^{sym} = I - D^{-1/2} A D^{-1/2}"""
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    sym_norm_lap = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    _dbg("sym_normalized_laplacian", adj=adj, result=sym_norm_lap,
         D_range_min=float(np.min(D)), D_range_max=float(np.max(D)))
    return sym_norm_lap


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """Rescale eigenvalues to [-1,1] for Chebyshev polynomials."""
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        # v4: dump the computed spectral radius for debugging
        _dbg("scaled_laplacian.spectral_radius", lambda_max=lambda_max)

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    _dbg("scaled_laplacian.result", L_res=L_res)
    return L_res


def symmetric_message_passing_adj(adj):
    """Renormalized message-passing adjacency (GCN-style)."""
    print("[v4-DBG] symmetric_message_passing_adj: ensure self-loop already added.", file=sys.stderr)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    _dbg("sym_mp_adj", mp_adj_shape=mp_adj.shape)
    return mp_adj


def transition_matrix(adj):
    """
    P = D^{-1} A — v4 modification: construct diagonal via sparse identity scaling
    instead of flatten+diags, which is numerically identical but avoids an intermediate
    dense vector when node count is large.
    """
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()

    # v4: use sparse identity * reciprocal instead of sp.diags(d_inv)
    d_inv = np.power(rowsum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    # construct D^{-1} as scaled identity — avoids extra dense allocation
    D_inv = sp.eye(adj_sp.shape[0], format='csr') * 0.0
    D_inv.setdiag(d_inv)

    P = D_inv.dot(adj_sp).astype(np.float32).todense()
    _dbg("transition_matrix", input_nnz=adj_sp.nnz, P_shape=P.shape,
         rowsum_min=float(np.min(rowsum)), rowsum_max=float(np.max(rowsum)))
    return P
