"""
Mask — Penumbra变体
算法改动: Bernoulli概率掩码 替代 确定性邻接掩码
  原版: 直接用predefined adj做元素乘法
  Penumbra: 将predefined adj视为Bernoulli概率 p
           训练时: 按 p 采样二值掩码 (straight-through estimator)
           推理时: 直接用概率 p 做确定性加权
           温度参数控制采样的"硬度"
"""
import torch
import torch.nn as nn
from .... import _dbg


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 温度: 控制Bernoulli采样的硬度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _bernoulli_mask(self, index, adj):
        """Bernoulli概率掩码"""
        base_mask = self.mask[index]
        prob = torch.clamp(base_mask, 0.0, 1.0)
        prob = prob + torch.ones_like(prob) * 1e-7

        if self.training:
            # Straight-through Bernoulli:
            # forward用hard sample, backward走soft概率
            temp = torch.clamp(self.temperature, min=0.1)
            # Gumbel-ish: 用sigmoid(logit/temp)逼近
            logits = torch.log(prob / (1 - prob + 1e-7))
            uniform = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(uniform + 1e-7) + 1e-7)
            soft = torch.sigmoid((logits + gumbel) / temp)
            # STE: forward=hard, backward=soft
            hard = (soft > 0.5).float()
            mask_out = hard - soft.detach() + soft
        else:
            mask_out = prob

        return mask_out.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._bernoulli_mask(index, a))

        _dbg("mask.temperature",
             self.temperature, "graph")

        return result
