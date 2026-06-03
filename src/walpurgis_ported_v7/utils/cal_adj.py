#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys

_DBG_ADJ = ("--dbg-adj" in sys.argv)


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
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    symmetric_normalized_laplacian = (
        sp.eye(adj.shape[0])
        - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo())

    if _DBG_ADJ:
        L = symmetric_normalized_laplacian.toarray()
        print(f"[DBG-ADJ] sym_norm_lap  shape={L.shape}  "
              f"nnz={symmetric_normalized_laplacian.nnz}  "
              f"range=[{L.min():.4f}, {L.max():.4f}]")

    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]

    if _DBG_ADJ:
        print(f"[DBG-ADJ] scaled_lap  lambda_max={lambda_max:.4f}")

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    return L_res


def symmetric_message_passing_adj(adj):
    print("calculating renormalized message passing adj, "
          "ensure self-loop has been added.")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = (d_mat_inv_sqrt.dot(adj).transpose()
              .dot(d_mat_inv_sqrt).astype(np.float32).todense())
    return mp_adj


def transition_matrix(adj):
    """算法改动: 加 epsilon=1e-8 防止孤立节点的除零"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    # 加 epsilon
    d_inv = np.power(rowsum + 1e-8, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()

    if _DBG_ADJ:
        print(f"[DBG-ADJ] transition_matrix  shape={P.shape}  "
              f"row_sum_range=[{np.array(P.sum(1)).min():.4f}, "
              f"{np.array(P.sum(1)).max():.4f}]  "
              f"isolated_nodes={(rowsum < 1e-6).sum()}")

    return P
