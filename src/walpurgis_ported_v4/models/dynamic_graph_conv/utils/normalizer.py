"""
Normalizer & MultiOrder — walpurgis_ported_v4
Modifications:
  - Normalizer._norm: changed from row normalization (D^{-1}A) to symmetric
    normalization (D^{-1/2} A D^{-1/2}) for better spectral properties
  - MultiOrder._multi_order: prints k-th order graph density for debugging
"""
import torch
import torch.nn as nn
import sys

from utils.cal_adj import remove_nan_inf

_V4_DEBUG = True


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        """v4: symmetric normalization D^{-1/2} A D^{-1/2} instead of D^{-1} A."""
        degree = torch.sum(graph, dim=2)
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))  # v4: sqrt for symmetric
        D_left = torch.diag_embed(d_inv_sqrt)
        # D^{-1/2} A D^{-1/2}
        normed_graph = torch.bmm(torch.bmm(D_left, graph), D_left)

        if _V4_DEBUG:
            density = (graph > 1e-6).float().mean().item()
            print(f"[v4-DBG][Normalizer._norm] "
                  f"graph_shape={tuple(graph.shape)} "
                  f"density={density:.4f} "
                  f"degree_range=[{degree.min().item():.4f},{degree.max().item():.4f}]",
                  file=sys.stderr)
        return normed_graph

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

            if _V4_DEBUG:
                density = (k_1_order > 1e-6).float().mean().item()
                print(f"[v4-DBG][MultiOrder] k={k} "
                      f"density={density:.4f} "
                      f"norm={k_1_order.norm().item():.4f}",
                      file=sys.stderr)

        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
