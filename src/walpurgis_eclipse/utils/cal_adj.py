"""Eclipse cal_adj: enhanced nan/inf reporting."""
import scipy.sparse as sp, numpy as np
from scipy.sparse import linalg
import torch, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor)); inf = torch.any(torch.isinf(tensor))
    if _ECL_DBG and (nan or inf):
        nan_idx = torch.where(torch.isnan(tensor))
        inf_idx = torch.where(torch.isinf(tensor))
        print(f"[ECL:cal_adj] NaN locations: {[x[:5].tolist() for x in nan_idx]}", file=sys.stderr)
        print(f"[ECL:cal_adj] Inf locations: {[x[:5].tolist() for x in inf_idx]}", file=sys.stderr)
    if raise_ex and (nan or inf): raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf

def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj); D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten(); D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat = sp.diags(D_inv_sqrt)
    return sp.eye(adj.shape[0]) - D_mat.dot(adj).dot(D_mat).tocoo()

def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected: adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None: lambda_max = linalg.eigsh(L, 1, which='LM')[0][0]
    L = sp.csr_matrix(L); M = L.shape[0]; I = sp.identity(M, format='csr', dtype=L.dtype)
    return (2 / lambda_max * L) - I

def symmetric_message_passing_adj(adj):
    adj = sp.coo_matrix(adj); rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten(); d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).transpose().dot(d_mat).astype(np.float32).todense()

def transition_matrix(adj):
    adj = sp.coo_matrix(adj); rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten(); d_inv[np.isinf(d_inv)] = 0.
    return sp.diags(d_inv).dot(adj).astype(np.float32).todense()
