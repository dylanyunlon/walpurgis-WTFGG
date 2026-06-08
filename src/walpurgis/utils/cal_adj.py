#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Cascade cal_adj: adjacency matrix utilities with cascade diagnostics."""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'

def check_nan_inf(tensor, raise_ex=True):
    # nan
    nan = torch.any(torch.isnan(tensor))
    # inf
    inf = torch.any(torch.isinf(tensor))
    if _CAS_DBG and (nan or inf):
        nan_ct = torch.isnan(tensor).sum().item()
        inf_ct = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        print(f"[CAS:cal_adj] NaN={nan_ct}/{total} ({nan_ct/total:.2%}) "
              f"Inf={inf_ct}/{total} ({inf_ct/total:.2%})", file=sys.stderr)
    # raise
    if raise_ex and (nan or inf):
        raise Exception({"nan":nan, "inf":inf})
    return {"nan":nan, "inf":inf}, nan or inf

def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def calculate_symmetric_normalized_laplacian(adj):
    adj                                 = sp.coo_matrix(adj)
    D                                   = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()    # diagonals of D^{-1/2}
    D_inv_sqrt[np.isinf(D_inv_sqrt)]    = 0.
    matrix_D_inv_sqrt                   = sp.diags(D_inv_sqrt)   # D^{-1/2}
    symmetric_normalized_laplacian      = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    return symmetric_normalized_laplacian

def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L       = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L       = sp.csr_matrix(L)
    M, _    = L.shape
    I       = sp.identity(M, format='csr', dtype=L.dtype)
    L_res   = (2 / lambda_max * L) - I
    return L_res

def symmetric_message_passing_adj(adj):
    adj         = sp.coo_matrix(adj)
    rowsum      = np.array(adj.sum(1))
    d_inv_sqrt  = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt  = sp.diags(d_inv_sqrt)
    mp_adj          = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj

def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P
