import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
from walpurgis import _dbg

_TAG = "adj"


def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))
    _dbg(_TAG, "check_nan_inf", has_nan=nan, has_inf=inf)
    if raise_ex and (nan or inf):
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


# ---- 改动1: RBF kernel ----
# upstream 直接用原始adj权重; 这里对距离型adj施加RBF变换
# σ自适应设为非零距离的中位数
def _rbf_kernel(adj_np):
    """对非零元素施加 exp(-d^2 / (2σ^2)), σ = median(非零元素)."""
    nz = adj_np[adj_np > 0]
    if len(nz) == 0:
        return adj_np
    sigma = float(np.median(nz))
    if sigma < 1e-10:
        sigma = 1.0
    rbf = np.exp(-adj_np ** 2 / (2.0 * sigma ** 2))
    # 保持零元素为零
    rbf = np.where(adj_np > 0, rbf, 0.0)
    print(f"[v10 cal_adj] RBF kernel: σ={sigma:.4f}, "
          f"nnz_before={np.count_nonzero(adj_np)}, "
          f"rbf_mean={rbf[rbf>0].mean():.4f}")
    return rbf


# ---- 改动2: k-NN 稀疏化 ----
# upstream 不做稀疏化, 保留全部边; 这里只保留每行 top-k 邻居
_KNN_K = 15


def _knn_sparsify(adj_np, k=_KNN_K):
    """每个节点只保留 top-k 最强连接."""
    n = adj_np.shape[0]
    sparse_adj = np.zeros_like(adj_np)
    for i in range(n):
        row = adj_np[i]
        if np.count_nonzero(row) <= k:
            sparse_adj[i] = row
        else:
            topk_idx = np.argpartition(row, -k)[-k:]
            sparse_adj[i, topk_idx] = row[topk_idx]
    kept = np.count_nonzero(sparse_adj)
    total = np.count_nonzero(adj_np)
    print(f"[v10 cal_adj] k-NN sparsify: k={k}, "
          f"edges {total} → {kept} ({100*kept/max(total,1):.1f}%)")
    return sparse_adj


# ---- 改动3: 双向对称闭包 ----
def _symmetric_closure(adj_np):
    """max(A, A^T) 保证对称."""
    sym = np.maximum(adj_np, adj_np.T)
    return sym


def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    matrix_D_inv_sqrt = sp.diags(D_inv_sqrt)
    L = (sp.eye(adj.shape[0])
         - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo())
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
    return L_res


def symmetric_message_passing_adj(adj):
    print("[v10] calc renormalized message passing adj "
          "(ensure self-loop added)")
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp = (d_mat_inv_sqrt.dot(adj).transpose()
          .dot(d_mat_inv_sqrt).astype(np.float32).todense())
    return mp


# ---- 改动4: epsilon-smooth transition ----
# upstream: d_inv 对零行直接设 0, 可能导致全零行
# v10: 加 epsilon 到每行, 保证不出现全零概率行
_TRANS_EPS = 1e-8


def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    # 改动4: epsilon smooth — upstream 直接 power(-1) 然后 inf→0
    rowsum = rowsum + _TRANS_EPS
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P
