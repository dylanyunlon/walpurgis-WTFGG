"""
cal_adj — Nightfall变体
算法改写:
  1. transition_matrix: 加eps防护防止除零
  2. 新增 gaussian_kernel_adj(): 用高斯核对邻接矩阵加权 (PEMS04/08)
  3. 所有函数加数值稳定性检查
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
from .. import _dbg


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
    D_inv_sqrt = np.power(D + 1e-10, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    symmetric_normalized_laplacian = (
        sp.eye(adj.shape[0]) -
        matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo())
    return symmetric_normalized_laplacian


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
    return L_res


def symmetric_message_passing_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum + 1e-10, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj


def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum + 1e-10, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P


def gaussian_kernel_adj(adj, sigma=1.0):
    """高斯核加权邻接矩阵: w_ij = exp(-d_ij^2 / (2*sigma^2))
    对非零元素做高斯衰减, 让远距离边权重更小"""
    nonzero_mask = adj > 0
    weighted = np.exp(-(adj ** 2) / (2 * sigma ** 2 + 1e-10))
    weighted = weighted * nonzero_mask
    np.fill_diagonal(weighted, 0)
    _dbg("cal_adj.gaussian", weighted, "data")
    return weighted.astype(np.float32)
