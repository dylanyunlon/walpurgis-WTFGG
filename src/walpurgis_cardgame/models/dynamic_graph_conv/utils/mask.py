"""
mask.py — CardGame Mask
算法改写 (vs upstream):
  - 硬mask (0/1乘法) → sigmoid soft-gating
  - 训练时加Gumbel noise实现可微分的随机mask
  - 推理时使用确定性sigmoid
"""
import os
import sys
import torch
import torch.nn as nn

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="Mask"):
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


class Mask(nn.Module):
    """CardGame Mask: sigmoid soft-gating + Gumbel noise

    改写:
      - 原始: hard mask (adj * binary_mask)
      - CardGame: sigmoid(logits + gumbel_noise) * adj
      - logits从predefined adj学习, Gumbel noise提供可微随机性
    """

    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 可学习的mask温度
        self.gumbel_temperature = nn.Parameter(torch.ones(1) * 2.0)

    def _gumbel_sample(self, shape, device, eps=1e-20):
        """采样Gumbel噪声"""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def _soft_mask(self, index, adj):
        """Sigmoid soft-gating with Gumbel noise"""
        base_mask = self.mask[index].to(adj.device)
        # 将base_mask转为logits (log(p/(1-p)))
        logits = torch.log(base_mask + 1e-7) - torch.log(1 - base_mask + 1e-7)

        if self.training:
            # Gumbel-sigmoid
            noise = self._gumbel_sample(logits.shape, logits.device)
            tau = torch.clamp(self.gumbel_temperature, min=0.1, max=10.0)
            soft_gate = torch.sigmoid((logits + noise) / tau)
        else:
            soft_gate = torch.sigmoid(logits)

        _dbg(f"soft_gate[{index}]", soft_gate)
        return soft_gate.unsqueeze(0) * adj

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._soft_mask(index, a))
        return result
