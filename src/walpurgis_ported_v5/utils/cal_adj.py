import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

# Delta vs upstream:
#   1. remove_nan_inf also clamps extreme values (|x|>1e6)
#   2. transition_matrix adds Laplacian smoothing (α=0.01)


def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))
    if raise_ex and (nan or inf):
        # ── debug dump before crash ──
        print(f"\033[91m[check_nan_inf] shape={tensor.shape} "
              f"nan={nan.item()} inf={inf.item()}\033[0m")
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    # ── delta 1: clamp extremes ──
    tensor = torch.clamp(tensor, min=-1e6, max=1e6)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    D   = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    sym_lap = (sp.eye(adj.shape[0]) -
               matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo())
    return sym_lap


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L   = sp.csr_matrix(L)
    M   = L.shape[0]
    I   = sp.identity(M, format='csr', dtype=L.dtype)
    L_res = (2 / lambda_max * L) - I
    return L_res


def symmetric_message_passing_adj(adj):
    print("calculating renormalized MP adj — ensure self-loop already added.")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    mp_adj = (d_mat.dot(adj).transpose().dot(d_mat)
              .astype(np.float32).todense())
    return mp_adj


def transition_matrix(adj):
    # ── delta 2: Laplacian smoothing ──
    alpha = 0.01
    adj_smooth = adj + alpha * np.eye(adj.shape[0])
    adj_sp = sp.coo_matrix(adj_smooth)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj_sp).astype(np.float32).todense()
    return P
