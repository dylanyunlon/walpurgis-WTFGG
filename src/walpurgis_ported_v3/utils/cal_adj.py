"""
Adjacency matrix algebra: Laplacians, transition matrices, nan/inf guards.
Ported with debug probes.
"""
import sys
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

_DBG = ("--debug-adj" in sys.argv)


# ─── Numeric guards ───

def check_nan_inf(tensor, raise_ex=True):
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    info = {"nan": has_nan.item(), "inf": has_inf.item()}
    if _DBG:
        print(f"[DBG:adj] check_nan_inf  shape={tuple(tensor.shape)}  {info}")
    if raise_ex and (has_nan or has_inf):
        raise ValueError(f"Tensor has nan/inf: {info}")
    return info, (has_nan or has_inf)


def remove_nan_inf(tensor):
    cleaned = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    cleaned = torch.where(torch.isinf(cleaned), torch.zeros_like(cleaned), cleaned)
    if _DBG:
        n_fixed = int((tensor != cleaned).sum().item())
        print(f"[DBG:adj] remove_nan_inf  fixed {n_fixed} entries  "
              f"shape={tuple(tensor.shape)}")
    return cleaned


# ─── Laplacians ───

def calculate_symmetric_normalized_laplacian(adj):
    """L^{sym} = I - D^{-1/2} A D^{-1/2}"""
    coo = sp.coo_matrix(adj)
    deg = np.array(coo.sum(1))
    d_isqrt = np.power(deg, -0.5).flatten()
    d_isqrt[np.isinf(d_isqrt)] = 0.0
    D_isqrt = sp.diags(d_isqrt)
    lap = sp.eye(coo.shape[0]) - D_isqrt.dot(coo).dot(D_isqrt).tocoo()
    if _DBG:
        print(f"[DBG:adj] sym_norm_lap  N={coo.shape[0]}  "
              f"nnz_lap={lap.nnz}  deg_range=[{deg.min():.2f},{deg.max():.2f}]")
    return lap


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """L_scaled = (2/λ_max) L - I   (for Chebyshev convolutions)"""
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    sym_L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(sym_L, 1, which='LM')
        lambda_max = lambda_max[0]
    csr_L = sp.csr_matrix(sym_L)
    n = csr_L.shape[0]
    eye = sp.identity(n, format='csr', dtype=csr_L.dtype)
    scaled = (2.0 / lambda_max * csr_L) - eye
    if _DBG:
        print(f"[DBG:adj] scaled_lap  λ_max={lambda_max:.4f}  N={n}")
    return scaled


# ─── Message-passing adjacency ───

def symmetric_message_passing_adj(adj):
    """D^{-1/2} A D^{-1/2}  (GCN renormalization)."""
    if _DBG:
        print("[DBG:adj] sym_mp_adj: ensure self-loop is already present")
    coo = sp.coo_matrix(adj)
    row_sum = np.array(coo.sum(1))
    d_isqrt = np.power(row_sum, -0.5).flatten()
    d_isqrt[np.isinf(d_isqrt)] = 0.0
    D_isqrt = sp.diags(d_isqrt)
    result = D_isqrt.dot(coo).transpose().dot(D_isqrt).astype(np.float32).todense()
    return result


def transition_matrix(adj):
    """Row-stochastic transition: P = D^{-1} A"""
    coo = sp.coo_matrix(adj)
    row_sum = np.array(coo.sum(1)).flatten()
    d_inv = np.power(row_sum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    D_inv = sp.diags(d_inv)
    P = D_inv.dot(coo).astype(np.float32).todense()
    if _DBG:
        print(f"[DBG:adj] transition_matrix  N={coo.shape[0]}  "
              f"row_sum range=[{row_sum.min():.2f},{row_sum.max():.2f}]  "
              f"P range=[{np.asarray(P).min():.4f},{np.asarray(P).max():.4f}]")
    return P
