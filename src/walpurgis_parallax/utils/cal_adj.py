"""
cal_adj — Parallax变体 (M054)
邻接矩阵计算: 对称归一化拉普拉斯, 缩放拉普拉斯,
对称消息传递, 转移矩阵
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg


def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    return (sp.eye(adj.shape[0])
            - matrix_D_inv_sqrt.dot(adj).dot(
                matrix_D_inv_sqrt).tocoo())


def calculate_scaled_laplacian(adj, lambda_max=2,
                               undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    return (2 / lambda_max * L) - I


def symmetric_message_passing_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32).todense())


def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
