import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

_TAG = "normalizer"


def _remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    """upstream: D^{-1}A row normalization
    改动: Sinkhorn-style doubly-stochastic normalization (3 iterations)
    Doubly-stochastic matrix: 行和=1且列和=1, 比单纯行归一化更对称
    交替做行归一化和列归一化, 3次迭代就很接近doubly-stochastic
    """

    def __init__(self):
        super().__init__()
        self.sinkhorn_iters = 3

    def _sinkhorn_norm(self, graph):
        """Sinkhorn迭代: 交替行/列归一化"""
        g = graph.clamp(min=0) + 1e-8
        for _ in range(self.sinkhorn_iters):
            # row normalization
            row_sum = g.sum(dim=-1, keepdim=True)
            g = g / _remove_nan_inf(row_sum)
            # column normalization
            col_sum = g.sum(dim=-2, keepdim=True)
            g = g / _remove_nan_inf(col_sum)
        # final row normalization for row-stochastic
        row_sum = g.sum(dim=-1, keepdim=True)
        g = g / _remove_nan_inf(row_sum)
        _dbg(f"{_TAG}/sinkhorn_row_sum",
             g.sum(dim=-1).mean(), _TAG)
        return g

    def forward(self, adj):
        return [self._sinkhorn_norm(a) for a in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        mask = 1 - torch.eye(graph.shape[-2]).to(graph.device)
        if graph.dim() == 2:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.order + 1):
                k_1_order = torch.matmul(k_1_order, graph)
                graph_ordered.append(k_1_order * mask)
        else:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask.unsqueeze(0))
            for k in range(2, self.order + 1):
                k_1_order = torch.bmm(k_1_order, graph)
                graph_ordered.append(k_1_order * mask.unsqueeze(0))
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
