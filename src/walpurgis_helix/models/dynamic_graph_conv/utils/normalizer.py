import torch
import torch.nn as nn

from walpurgis_helix.utils.cal_adj import remove_nan_inf


class Normalizer(nn.Module):
    """Helix改写: 加入对称归一化选项,
    D^{-1/2} A D^{-1/2} 替代 D^{-1} A"""
    def __init__(self, symmetric=True):
        super().__init__()
        self.symmetric = symmetric

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        if self.symmetric:
            # Helix: 对称归一化 D^{-1/2} A D^{-1/2}
            degree_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
            degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
            normed_graph = torch.bmm(torch.bmm(degree_inv_sqrt, graph), degree_inv_sqrt)
        else:
            degree  = remove_nan_inf(1 / degree)
            degree  = torch.diag_embed(degree)
            normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]

class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order  = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph               # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order+1):     # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
