"""Flux Normalizer: 对称归一化 D^{-1/2}AD^{-1/2} + MultiOrder.
与upstream(row-norm D^{-1}A)和vortex(Sinkhorn doubly stochastic)不同,
Flux使用对称归一化, 保留图拉普拉斯的谱性质,
适合流式推理中需要稳定的图卷积操作."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    """对称归一化: D^{-1/2} A D^{-1/2}
    比row-normalization更保留谱结构,
    比Sinkhorn计算更轻量."""
    def __init__(self):
        super().__init__()

    def _symmetric_norm(self, graph):
        # D^{-1/2}
        degree = torch.sum(graph, dim=2)
        degree_inv_sqrt = _remove_nan_inf(
            1.0 / torch.sqrt(degree + 1e-8))
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        # D^{-1/2} A D^{-1/2}
        normed = torch.bmm(
            torch.bmm(D_inv_sqrt, graph), D_inv_sqrt)
        normed = _remove_nan_inf(normed)
        if _FX_DBG:
            eigenvalues_approx = torch.sum(
                normed, dim=-1).mean().item()
            print(f"[FX:sym_norm@normalizer] "
                  f"degree_mean="
                  f"{degree.mean().item():.4f} "
                  f"norm_row_sum_mean="
                  f"{eigenvalues_approx:.4f}",
                  file=sys.stderr)
        return normed

    def forward(self, adj):
        return [self._symmetric_norm(a) for a in adj]


class MultiOrder(nn.Module):
    """Multi-order graph expansion with standard power iteration."""
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = 1 - torch.eye(
            graph.shape[1]).to(graph.device)
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
