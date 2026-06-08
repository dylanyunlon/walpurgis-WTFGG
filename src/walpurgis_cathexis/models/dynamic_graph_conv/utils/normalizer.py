"""
Cathexis Normalizer — 算法改写 #6
upstream: row-normalize D^{-1}A
cathexis: Sinkhorn iteration for doubly-stochastic normalization
"""
import torch
import torch.nn as nn
from ....utils.cal_adj import remove_nan_inf

class Normalizer(nn.Module):
    def __init__(self, sinkhorn_iters=5):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters

    def _sinkhorn_norm(self, graph):
        graph = graph.clamp(min=0)
        for _ in range(self.sinkhorn_iters):
            row_sum = graph.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            graph = graph / row_sum
            col_sum = graph.sum(dim=-2, keepdim=True).clamp(min=1e-8)
            graph = graph / col_sum
        return graph

    def forward(self, adj):
        return [self._sinkhorn_norm(a) for a in adj]

class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = 1 - torch.eye(graph.shape[1]).to(graph.device)
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order+1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
