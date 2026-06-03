import torch
import torch.nn as nn
from walpurgis import _dbg

_TAG = "norm"


def _remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    """改动1: 行归一化 D^{-1}A → 对称归一化 D^{-1/2} A D^{-1/2}.
    upstream 用单侧 D^{-1}A, 非对称; 对称归一化保证谱性质更好."""
    def __init__(self):
        super().__init__()

    def _symmetric_norm(self, graph):
        degree = torch.sum(graph, dim=-1)
        d_inv_sqrt = _remove_nan_inf(torch.pow(degree, -0.5))
        # D^{-1/2} 对角矩阵
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # D^{-1/2} A D^{-1/2}
        normed = torch.bmm(torch.bmm(D_inv_sqrt, graph), D_inv_sqrt)
        _dbg(_TAG, "sym_norm", degree_mean=degree.mean(), normed=normed)
        return normed

    def forward(self, adj):
        return [self._symmetric_norm(a) for a in adj]


# 改动2: 指数衰减 λ^k
_DECAY_LAMBDA = 0.8


class MultiOrder(nn.Module):
    """改动2: 高阶图乘积加指数衰减 λ^k.
    upstream 无衰减, 高阶权重和一阶相同; 这里让远距离邻居影响指数递减.
    改动3: 对角线 mask 的 eps 可学习."""
    def __init__(self, order=2):
        super().__init__()
        self.order = order
        # 改动3: 可学习 eps 控制对角线 mask 强度
        self.log_eps = nn.Parameter(torch.tensor(-6.0))  # init ~1e-6

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        eye = torch.eye(graph.shape[-1], device=graph.device)
        if graph.dim() == 3:
            eye = eye.unsqueeze(0)
        mask = 1.0 - eye

        # 1阶: 无衰减
        graph_ordered.append(k_1_order * mask)

        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            # 改动2: 指数衰减
            decay = _DECAY_LAMBDA ** (k - 1)
            masked_k = k_1_order * mask * decay
            graph_ordered.append(masked_k)
            _dbg(_TAG, f"order_{k}", decay=decay, masked_k=masked_k)

        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
