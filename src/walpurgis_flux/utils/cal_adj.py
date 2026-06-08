"""Flux cal_adj: 频谱诊断 + 数值安全.
与upstream(基础check_nan_inf)和vortex(percentile reporting)不同,
Flux加入频谱诊断: 检查adj矩阵的谱半径和条件数,
帮助判断图卷积的数值稳定性."""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))
    if _FX_DBG and (nan or inf):
        nan_ct = torch.isnan(tensor).sum().item()
        inf_ct = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        print(f"[FX:cal_adj] NaN={nan_ct}/{total} "
              f"({nan_ct/total:.2%}) "
              f"Inf={inf_ct}/{total} "
              f"({inf_ct/total:.2%})",
              file=sys.stderr)
        # Flux: 频谱诊断 — 对2D tensor计算谱半径估计
        valid = tensor[~torch.isnan(tensor) &
                       ~torch.isinf(tensor)]
        if valid.numel() > 0 and tensor.dim() == 2:
            try:
                sv = torch.linalg.svdvals(
                    tensor.float().nan_to_num())
                spectral_radius = sv[0].item()
                cond_number = (sv[0] / sv[-1].clamp(
                    min=1e-10)).item()
                print(f"[FX:cal_adj] spectral_radius="
                      f"{spectral_radius:.4f} "
                      f"cond_number={cond_number:.2f}",
                      file=sys.stderr)
            except Exception:
                pass
    if raise_ex and (nan or inf):
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf


def remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    D = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat = sp.diags(D_inv_sqrt)
    return (sp.eye(adj.shape[0]) -
            D_mat.dot(adj).dot(D_mat).tocoo())


def calculate_scaled_laplacian(adj, lambda_max=2,
                               undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max = linalg.eigsh(L, 1, which='LM')[0][0]
    L = sp.csr_matrix(L)
    M = L.shape[0]
    I = sp.identity(M, format='csr', dtype=L.dtype)
    return (2 / lambda_max * L) - I


def symmetric_message_passing_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat.dot(adj).transpose()
            .dot(d_mat).astype(np.float32).todense())


def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    return (sp.diags(d_inv).dot(adj)
            .astype(np.float32).todense())
