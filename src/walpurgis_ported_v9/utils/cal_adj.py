#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
cal_adj.py  — v9 port
======================
Upstream delta (≈20 %):
  1. remove_nan_inf() → torch.nan_to_num  (PyTorch-native, no branch)
  2. symmetric_normalized_laplacian adds epsilon=1e-8 before D^{-1/2}
     to guard isolated nodes
  3. transition_matrix gains `self_loop` kwarg  (default False to stay
     backward-compatible, but handy during ablation)
  4. Dense _dbg() checkpoints after every matrix computation
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

from walpurgis_ported_v9 import _dbg

_TAG = "cal_adj"

# ────────────────────────── numeric guard ──────────────────────────

def check_nan_inf(tensor, raise_ex=True):
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    report = {"nan": has_nan.item(), "inf": has_inf.item()}
    _dbg(_TAG, f"check_nan_inf → {report}")
    if raise_ex and (has_nan or has_inf):
        raise ValueError(f"Numeric fault detected: {report}")
    return report, (has_nan or has_inf)


def remove_nan_inf(tensor):
    # v9: single-call nan_to_num replaces two separate torch.where passes
    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    _dbg(_TAG, f"remove_nan_inf  in_abs_max={tensor.abs().max().item():.6g}  "
               f"out_abs_max={cleaned.abs().max().item():.6g}")
    return cleaned

# ──────────────────── Laplacian utilities ──────────────────────────

_EPS_LAPLACIAN = 1e-8        # v9: epsilon regularisation


def calculate_symmetric_normalized_laplacian(adj):
    """
    L^{Sym} = I - D^{-1/2} A D^{-1/2}

    v9 change: adds *_EPS_LAPLACIAN* to degree before inversion so that
    isolated nodes (degree == 0) receive a finite self-loop contribution
    rather than producing inf / NaN.
    """
    adj_sp = sp.coo_matrix(adj)
    degree = np.array(adj_sp.sum(1)).flatten()

    # v9: epsilon guard ─ the key algorithmic tweak
    degree_safe = degree + _EPS_LAPLACIAN
    d_inv_sqrt = np.power(degree_safe, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0          # belt-and-suspenders

    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_sym = sp.eye(adj_sp.shape[0]) - D_inv_sqrt.dot(adj_sp).dot(D_inv_sqrt).tocoo()

    _dbg(_TAG, f"sym_norm_laplacian  shape={L_sym.shape}  nnz={L_sym.nnz}  "
               f"degree_min={degree.min():.4g}  degree_max={degree.max():.4g}")
    return L_sym


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """
    L_scaled = (2/λ_max) L^{Sym} − I

    Preserves upstream logic; debug prints eigenvalue.
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        _dbg(_TAG, f"scaled_laplacian  computed λ_max={lambda_max:.6g}")
    else:
        _dbg(_TAG, f"scaled_laplacian  fixed λ_max={lambda_max}")

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2.0 / lambda_max * L) - I
    return L_res


def symmetric_message_passing_adj(adj):
    """D^{-1/2} A D^{-1/2}  (renormalised GCN)."""
    _dbg(_TAG, "symmetric_message_passing_adj  — ensure self-loop already added")
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = D_inv_sqrt.dot(adj_sp).transpose().dot(D_inv_sqrt).astype(np.float32).todense()
    _dbg(_TAG, f"mp_adj  shape={mp_adj.shape}  "
               f"min={np.min(mp_adj):.6g}  max={np.max(mp_adj):.6g}")
    return mp_adj


def transition_matrix(adj, self_loop: bool = False):
    """
    P = D^{-1} A

    v9 change: optional *self_loop* injection (A ← A + I) before
    computing the transition matrix, controllable per-call.
    """
    if self_loop:
        adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)
        _dbg(_TAG, "transition_matrix  self-loop injected")

    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv = np.power(rowsum, -1)
    d_inv[np.isinf(d_inv)] = 0.0
    D_inv = sp.diags(d_inv)
    P = D_inv.dot(adj_sp).astype(np.float32).todense()
    _dbg(_TAG, f"transition_matrix  shape={P.shape}  "
               f"rowsum_min={rowsum.min():.4g}  rowsum_max={rowsum.max():.4g}")
    return P
