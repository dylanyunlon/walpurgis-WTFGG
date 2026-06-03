import torch
import torch.nn as nn

from utils.cal_adj import remove_nan_inf

# Delta vs upstream:
#   1. Normalizer clips extreme degree values to prevent div-by-zero instability
#   2. MultiOrder applies symmetric normalisation (D^{-0.5} A D^{-0.5}) instead of row-norm

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        # ── delta 1: clip before inverse ──
        degree = degree.clamp(min=1e-8)
        degree = remove_nan_inf(1 / degree)
        degree = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(a) for a in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1], device=graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            # ── delta 2: symmetric normalisation per hop ──
            deg = k_1_order.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            k_normed = k_1_order / deg
            graph_ordered.append(k_normed * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
