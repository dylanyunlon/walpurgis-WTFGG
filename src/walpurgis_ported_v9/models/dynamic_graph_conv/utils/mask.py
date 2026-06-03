"""
mask.py — v9 port
Algo delta:
  1. upstream 硬二值: mask = predefined_adj + 1e-7, 再 element-wise 乘
     → v9: 可学习 soft-threshold sigmoid:
       mask = σ(α · (predefined_adj − threshold))
     α 可训练 (init=10.0), threshold 固定 0.0
     → 连续可微, 允许梯度流过 mask 结构
  2. 对角线显式清零 (防自环干扰动态图)
"""
import torch
import torch.nn as nn
from walpurgis_ported_v9 import _dbg

_TAG = "mask"


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.predefined = model_args['adjs']
        # v9: learnable sharpness for soft mask
        self.alpha = nn.Parameter(torch.tensor(10.0))

    def _soft_mask(self, index, adj):
        base = self.predefined[index].to(adj.device)
        # v9: sigmoid soft-threshold
        soft = torch.sigmoid(self.alpha * base)
        # v9: zero-out diagonal
        diag_mask = 1.0 - torch.eye(soft.shape[-1], device=adj.device)
        soft = soft * diag_mask
        return soft * adj

    def forward(self, adj):
        result = []
        for idx, a in enumerate(adj):
            masked = self._soft_mask(idx, a)
            result.append(masked)
        _dbg(_TAG, f"soft-mask  α={self.alpha.item():.3f}  "
                    f"sparsity={[(1-(r>1e-6).float().mean().item())*100 for r in result]}")
        return result
