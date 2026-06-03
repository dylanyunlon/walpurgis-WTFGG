"""Adjacency matrix pre-processing utilities.

Algorithm changes vs upstream
-----------------------------
1. ``remove_nan_inf`` — adds an *eps-clamp* branch: instead of zeroing out
   inf values (which destroys degree information), clamp them to ±1e6.
   A flag ``hard_zero`` preserves the old behaviour when needed.
2. ``calculate_scaled_laplacian`` — eigenvalue computation uses a *shift-
   invert* sigma hint so ARPACK converges faster on denser graphs.
3. ``transition_matrix`` — detects near-isolated nodes (degree < 1e-8)
   and assigns them a self-loop weight of 1.0 instead of leaving 0/inf.
"""

import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch


_EPS_CLAMP = 1e6


def check_nan_inf(tensor, raise_ex=True):
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    info = {"nan": has_nan, "inf": has_inf}
    if raise_ex and (has_nan or has_inf):
        raise ValueError(f"Tensor health check failed: {info}")
    return info, has_nan or has_inf


def remove_nan_inf(tensor, hard_zero=False):
    """Replace nan/inf.  Default: clamp to ±_EPS_CLAMP (preserves sign).
    Set hard_zero=True to get the upstream zero-out behaviour."""
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    if hard_zero:
        tensor = torch.where(torch.isinf(tensor),
                             torch.zeros_like(tensor), tensor)
    else:
        tensor = torch.clamp(tensor, min=-_EPS_CLAMP, max=_EPS_CLAMP)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    deg = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    L_sym = sp.eye(adj.shape[0]) - D.dot(adj).dot(D).tocoo()
    return L_sym


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        # shift-invert mode for faster ARPACK convergence on dense graphs
        try:
            lambda_max, _ = linalg.eigsh(L, 1, which='LM', sigma=1.0)
        except linalg.ArpackNoConvergence:
            lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_scaled = (2.0 / lambda_max * L) - I
    return L_scaled


def symmetric_message_passing_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    mp = D.dot(adj).transpose().dot(D).astype(np.float32).todense()
    return mp


def transition_matrix(adj):
    """Row-normalised transition matrix P = D^{-1} A.

    Isolated-node guard: nodes with near-zero degree get a self-loop
    injected *before* normalisation, so P never contains inf rows.
    """
    adj = sp.coo_matrix(adj).astype(np.float64)
    rowsum = np.array(adj.sum(1)).flatten()
    # detect near-isolated nodes and inject self-loop
    isolated = rowsum < 1e-8
    if np.any(isolated):
        diag_fix = sp.diags(isolated.astype(np.float64))
        adj = adj + diag_fix
        rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    D = sp.diags(d_inv)
    P = D.dot(adj).astype(np.float32).todense()
    return P
