"""
Normalizer — Nightfall变体
算法改写: 行归一化 D^{-1}A → 对称归一化 D^{-1/2}AD^{-1/2}
对称归一化保持图谱的对称性, 与GCN理论一致
"""
import torch
import torch.nn as nn
from ....utils.cal_adj import remove_nan_inf
from .... import _dbg


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _sym_norm(self, graph):
        """对称归一化: D^{-1/2} A D^{-1/2}"""
        degree = torch.sum(graph, dim=2)
        deg_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        D_left = torch.diag_embed(deg_inv_sqrt)
        D_right = torch.diag_embed(deg_inv_sqrt)
        normed = torch.bmm(torch.bmm(D_left, graph), D_right)
        _dbg("normalizer.sym_norm", normed, "model")
        return normed

    def forward(self, adj):
        return [self._sym_norm(g) for g in adj]


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
        return [self._multi_order(a) for a in adj]
