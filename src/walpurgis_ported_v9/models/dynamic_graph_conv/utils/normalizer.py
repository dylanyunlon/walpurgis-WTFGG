"""
normalizer.py — v9 port
Algo delta:
  1. _norm: upstream 用 D^{-1}A (row-stochastic)
     → v9 用对称归一化 D^{-1/2} A D^{-1/2} (GCN 风格, 保持谱对称性)
  2. MultiOrder: 高阶邻接乘衰减因子 β^k (β=0.85), 抑制高阶远距离噪声
  3. 对角线掩码用 fill_diagonal_(0) 原地操作, 省掉矩阵乘法
"""
import torch
import torch.nn as nn
from utils.cal_adj import remove_nan_inf
from walpurgis_ported_v9 import _dbg

_TAG = "normalizer"

_DECAY_BETA = 0.85          # v9: high-order decay


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        # v9: symmetric normalisation D^{-1/2} A D^{-1/2}
        degree = torch.sum(graph, dim=2)
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        D = torch.diag_embed(d_inv_sqrt)
        normed = torch.bmm(torch.bmm(D, graph), D)
        _dbg(_TAG, f"sym_norm  degree∈[{degree.min().item():.3g},{degree.max().item():.3g}]")
        return normed

    def forward(self, adj):
        return [self._norm(a) for a in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        ordered = []
        k_pow = graph.clone()
        # v9: fill_diagonal_ instead of (1-eye)*
        k_1 = k_pow.clone()
        for b in range(k_1.shape[0]):
            k_1[b].fill_diagonal_(0)
        ordered.append(k_1)

        for k in range(2, self.order + 1):
            k_pow = torch.matmul(k_pow, graph)
            k_cur = k_pow.clone()
            for b in range(k_cur.shape[0]):
                k_cur[b].fill_diagonal_(0)
            # v9: decay factor β^k
            decay = _DECAY_BETA ** k
            ordered.append(k_cur * decay)
            _dbg(_TAG, f"multi_order k={k}  decay={decay:.4f}  "
                        f"|A^k|_max={k_cur.abs().max().item():.4g}")
        return ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
