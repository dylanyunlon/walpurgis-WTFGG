"""Meridian Normalizer — symmetric normalization D^{-1/2}AD^{-1/2}.
Changes vs upstream:
  - Symmetric normalization (upstream: row-only D^{-1}A)
  - MultiOrder with decay factor per hop (upstream: uniform weight)
"""
import torch
import torch.nn as nn
from walpurgis_meridian.utils.cal_adj import remove_nan_inf
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        # symmetric normalization: D^{-1/2} A D^{-1/2}
        degree = torch.sum(graph, dim=2)
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        d_left = torch.diag_embed(d_inv_sqrt)
        d_right = torch.diag_embed(d_inv_sqrt)
        normed_graph = torch.bmm(torch.bmm(d_left, graph), d_right)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order
        # learnable decay per hop
        self.decay = nn.Parameter(torch.tensor(0.8))

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        decay_val = torch.sigmoid(self.decay)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            weight = decay_val ** (k - 1)
            graph_ordered.append(k_1_order * mask * weight)
            if _DBG:
                print(f"[MER:multi_order] k={k} decay_weight={weight.item():.4f} "
                      f"graph_norm={k_1_order.detach().norm().item():.4f}", file=sys.stderr)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
