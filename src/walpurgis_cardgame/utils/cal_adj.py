"""
D2STGNN CardGame variant — cal_adj.py
Algorithm changes vs upstream:
  1. Added Cauchy kernel adjacency: A_cauchy(i,j) = gamma^2 / (gamma^2 + dist(i,j)^2)
  2. Added symmetric closure: A_sym = max(A, A^T) ensuring undirected graph
"""

import os
import sys
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        if isinstance(tensor, np.ndarray):
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min():.6f} max={tensor.max():.6f} "
                   f"mean={tensor.mean():.6f} std={tensor.std():.6f}")
        else:
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
                   f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))
    if raise_ex and (nan or inf):
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    r"""
    Calculate Symmetric Normalized Laplacian.
    L^{Sym} = I - D^{-1/2} A D^{-1/2}
    """
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    symmetric_normalized_laplacian = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    _dbg("cal_adj.sym_norm_lap.nnz", symmetric_normalized_laplacian.nnz, "cal_adj")
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    r"""
    Re-scaled the eigenvalue to [-1, 1] by scaling the normalized laplacian matrix for chebyshev pol.
    L_{scaled} = (2 / lambda_max * L) - I
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    return L_res


def symmetric_message_passing_adj(adj):
    r"""
    Calculate the renormalized message passing adj in GCN.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj


def transition_matrix(adj):
    r"""
    Calculate the transition matrix P = D^{-1}A
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P


# --- CARDGAME: Cauchy kernel adjacency ---
def cauchy_kernel_adjacency(adj, gamma=1.0):
    """Compute Cauchy kernel adjacency matrix.

    For each pair (i,j), the Cauchy kernel weight is:
        A_cauchy(i,j) = gamma^2 / (gamma^2 + dist(i,j)^2)

    where dist(i,j) is the original adjacency weight treated as distance.
    Non-zero entries in adj are treated as distances; zeros remain zero.

    Args:
        adj: np.ndarray, adjacency/distance matrix (N, N)
        gamma: float, Cauchy kernel bandwidth parameter

    Returns:
        cauchy_adj: np.ndarray, Cauchy kernel adjacency (N, N)
    """
    adj = np.array(adj, dtype=np.float64)
    mask = (adj != 0).astype(np.float64)
    gamma_sq = gamma ** 2
    cauchy_adj = (gamma_sq / (gamma_sq + adj ** 2)) * mask
    np.fill_diagonal(cauchy_adj, 0)  # no self-loops in Cauchy
    _dbg("cauchy_kernel.density", np.mean(cauchy_adj > 0), "cal_adj")
    _dbg("cauchy_kernel.mean_weight", np.mean(cauchy_adj[cauchy_adj > 0]) if np.any(cauchy_adj > 0) else 0.0, "cal_adj")
    return cauchy_adj.astype(np.float32)


# --- CARDGAME: symmetric closure ---
def symmetric_closure(adj):
    """Ensure adjacency is symmetric: A_sym = max(A, A^T)

    Args:
        adj: np.ndarray, possibly asymmetric adjacency (N, N)

    Returns:
        sym_adj: np.ndarray, symmetric adjacency (N, N)
    """
    adj = np.array(adj, dtype=np.float32)
    sym = np.maximum(adj, adj.T)
    _dbg("sym_closure.asymmetry_before", float(np.sum(np.abs(adj - adj.T))), "cal_adj")
    return sym
