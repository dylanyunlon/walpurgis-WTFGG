"""Graph normalizer + multi-order expansion.

Changes
-------
1. ``Normalizer`` — offers a ``symmetric`` mode (D^{-1/2} A D^{-1/2})
   alongside the default row-stochastic normalisation (D^{-1} A).
   Symmetric normalisation preserves eigenvalue properties which can
   stabilise spectral-based reasoning in the dynamic graph branch.
2. ``MultiOrder`` — the k-th order is computed via matrix power
   ``torch.linalg.matrix_power`` instead of sequential matmul in a loop.
   Semantically identical but lets cuBLAS batch the operation for k > 2.
"""

import torch
import torch.nn as nn
from utils.cal_adj import remove_nan_inf
from walpurgis_ported_v6 import _dbg


class Normalizer(nn.Module):
    def __init__(self, symmetric=False):
        super().__init__()
        self.symmetric = symmetric

    def _row_norm(self, graph):
        deg = torch.sum(graph, dim=2)
        deg = remove_nan_inf(1.0 / deg)
        D = torch.diag_embed(deg)
        return torch.bmm(D, graph)

    def _sym_norm(self, graph):
        deg = torch.sum(graph, dim=2)
        deg_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(deg + 1e-12))
        D = torch.diag_embed(deg_inv_sqrt)
        return torch.bmm(torch.bmm(D, graph), D)

    def forward(self, adj_list):
        fn = self._sym_norm if self.symmetric else self._row_norm
        result = [fn(a) for a in adj_list]
        _dbg("Normalizer", result[0], mode="sym" if self.symmetric else "row")
        return result


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _expand(self, graph):
        ordered = []
        mask = 1.0 - torch.eye(graph.shape[1], device=graph.device)
        for k in range(1, self.order + 1):
            if k == 1:
                g_k = graph
            else:
                # matrix power instead of sequential matmul
                g_k = torch.linalg.matrix_power(graph, k)
            ordered.append(g_k * mask)
        return ordered

    def forward(self, adj_list):
        return [self._expand(a) for a in adj_list]
