"""
Corona Normalizer — 算法改写:
  upstream: D^{-1}A (行归一化)
  corona: 交替归一化 — 先行归一化再列归一化, 平衡入度和出度影响
"""
import torch
import torch.nn as nn


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        # Corona改写: 交替行列归一化 D_row^{-1/2} A D_col^{-1/2}
        row_sum = torch.sum(graph, dim=2)
        row_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(row_sum + 1e-8))
        row_diag = torch.diag_embed(row_inv_sqrt)

        col_sum = torch.sum(graph, dim=1)
        col_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(col_sum + 1e-8))
        col_diag = torch.diag_embed(col_inv_sqrt)

        normed = torch.bmm(torch.bmm(row_diag, graph), col_diag)
        return normed

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
