import torch
import torch.nn as nn
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:norm:{tag}] row_sum_mean={val.sum(dim=-1).mean().item():.6f}", file=sys.stderr)

def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

class Normalizer(nn.Module):
    """upstream: D^-1 A 行归一化
    aurora/solstice: D^-1/2 A D^-1/2 对称归一化 (谱等价)"""
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        D_left = torch.diag_embed(d_inv_sqrt)
        D_right = torch.diag_embed(d_inv_sqrt)
        normed = torch.bmm(torch.bmm(D_left, graph), D_right)
        _sdbg("sym_norm", normed)
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
        mask = 1 - torch.eye(graph.shape[1]).to(graph.device)
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
