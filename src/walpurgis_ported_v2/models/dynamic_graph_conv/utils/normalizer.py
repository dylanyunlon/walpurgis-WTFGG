"""
Graph normalizer (row-stochastic) and multi-order expansion for
dynamic graph convolution.
"""

import torch
import torch.nn as nn
import sys

from utils.cal_adj import remove_nan_inf

_DBG_NORM = ("--debug-norm" in sys.argv) or False


class Normalizer(nn.Module):
    """Row-normalize each adjacency matrix: D^{-1} · A."""

    def __init__(self):
        super().__init__()

    def _row_normalize(self, graph):
        deg = torch.sum(graph, dim=2)
        deg_inv = remove_nan_inf(1.0 / deg)
        D_inv = torch.diag_embed(deg_inv)
        normed = torch.bmm(D_inv, graph)
        if _DBG_NORM:
            print(f"[DBG:norm] row_normalize  shape={tuple(graph.shape)}  "
                  f"deg_range=[{deg.min().item():.4f},{deg.max().item():.4f}]")
        return normed

    def forward(self, adj_list):
        return [self._row_normalize(a) for a in adj_list]


class MultiOrder(nn.Module):
    """Expand adjacency to k-th order powers (self-loop masked)."""

    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _expand_orders(self, graph):
        n = graph.shape[1]
        eye_mask = 1.0 - torch.eye(n, device=graph.device)
        orders = []
        power_k = graph
        orders.append(power_k * eye_mask)
        for k in range(2, self.order + 1):
            power_k = torch.matmul(power_k, graph)
            orders.append(power_k * eye_mask)
        if _DBG_NORM:
            norms_str = ", ".join(f"{o.norm().item():.4f}" for o in orders)
            print(f"[DBG:norm] multi_order  orders={self.order}  "
                  f"norms=[{norms_str}]")
        return orders

    def forward(self, adj_list):
        return [self._expand_orders(a) for a in adj_list]
