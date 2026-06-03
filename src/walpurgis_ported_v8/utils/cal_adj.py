import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys

_DBG = ("--dbg" in sys.argv)


def _dp(tag, msg):
    if _DBG:
        print(f"[DBG][cal_adj][{tag}] {msg}", flush=True)


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


def calculate_symmetric_normalized_laplacian(adj, eps=1e-8):
    """算法改动: D^{-1/2} 计算时加 eps 防止除零
    原版: np.power(D, -0.5) 后手动把 inf 置零
    改为: np.power(D + eps, -0.5), 数值连续性更好
    """
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D + eps, -0.5).flatten()
    _dp("sym_norm_lap", f"D range=[{D.min():.4f}, {D.max():.4f}]")
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    L = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(
        matrix_D_inv_sqrt).tocoo()
    return L


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
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
    _dp("scaled_lap", f"lambda_max={lambda_max:.4f}  shape={M}")
    return L_res


def symmetric_message_passing_adj(adj):
    print("calculating the renormalized message passing adj, "
          "please ensure that self-loop has added to adj.")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj


def transition_matrix(adj, row_clamp_min=1e-6):
    """算法改动: row-sum clamp
    原版: d_inv 中 inf 直接置零 → 孤立节点行全零
    改为: row_sum clamp 到 row_clamp_min, 保证即使近零行也有微弱的均匀转移
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    rowsum = np.maximum(rowsum, row_clamp_min)
    d_inv = np.power(rowsum, -1).flatten()
    _dp("transition_matrix",
        f"rowsum range=[{rowsum.min():.6f}, {rowsum.max():.6f}]  "
        f"clamped at {row_clamp_min}")
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P
