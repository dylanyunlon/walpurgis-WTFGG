"""
Mask — Nightfall变体
算法改写: 硬mask → 可学习sigmoid soft-gating (nn.ParameterList)
每个mask对应一个可训练的gate bias, 动态调节mask松紧
"""
import torch
import torch.nn as nn
from .... import _dbg


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 每个adjacency对应一个可学习gate bias
        self.gate_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in self.mask
        ])

    def _soft_mask(self, index, adj):
        base_mask = self.mask[index].to(adj.device)
        # sigmoid soft-gate: bias>0时mask更宽松, <0时更严格
        soft_gate = torch.sigmoid(base_mask + self.gate_biases[index])
        _dbg(f"mask.gate_{index}", self.gate_biases[index], "model")
        return soft_gate * adj

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._soft_mask(index, a))
        return result
