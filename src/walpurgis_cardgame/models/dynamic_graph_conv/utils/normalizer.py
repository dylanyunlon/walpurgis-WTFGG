"""
normalizer.py — CardGame Normalizer
算法改写 (vs upstream):
  - D⁻¹A (random walk归一化) → 拉普拉斯归一化 I - D^{-1/2}AD^{-1/2}
  - Laplacian归一化保留对称性, 特征值在[0,2], 有更好的谱性质
"""
import os
import sys
import torch
import torch.nn as nn

from walpurgis_cardgame.utils.cal_adj import remove_nan_inf

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="Normalizer"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class Normalizer(nn.Module):
    """CardGame Normalizer: 拉普拉斯归一化 I - D^{-1/2}AD^{-1/2}"""

    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        """拉普拉斯归一化: L_sym = I - D^{-1/2} A D^{-1/2}"""
        degree = torch.sum(graph, dim=2)
        # D^{-1/2}
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)
        # D^{-1/2} A D^{-1/2}
        normed = torch.bmm(torch.bmm(d_inv_sqrt_mat, graph), d_inv_sqrt_mat)
        # I - D^{-1/2}AD^{-1/2}
        identity = torch.eye(graph.shape[1], device=graph.device).unsqueeze(0)
        laplacian = identity - normed
        _dbg("laplacian_norm", laplacian)
        return laplacian

    def forward(self, adj):
        return [self._norm(a) for a in adj]


class MultiOrder(nn.Module):
    """多阶图卷积矩阵"""

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
