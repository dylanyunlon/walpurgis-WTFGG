"""Flux Mask: 温度调控的软门控掩码.
与upstream(固定binary mask)和vortex(straight-through Bernoulli)不同,
Flux使用温度参数控制sigmoid的锐度: 低温接近hard mask, 高温接近soft mask.
训练初期高温(soft, 梯度好流动), 后期低温(hard, 逼近稀疏图).
这种退火策略配合流式推理的graph构建需求."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Flux: 可学习温度参数 (初始高温=soft)
        self.temperature = nn.Parameter(
            torch.tensor(2.0))
        # 每个adj模板的可学习阈值偏置
        self.threshold_bias = nn.ParameterList([
            nn.Parameter(torch.zeros_like(a))
            for a in self.mask
        ])

    def _soft_mask(self, index, adj):
        """温度调控的软门控:
        score = sigmoid((adj_template + bias) / temperature)
        低温→接近hard mask, 高温→接近soft mask"""
        temp = torch.clamp(
            torch.abs(self.temperature), min=0.1)
        mask_template = self.mask[index].to(adj.device)
        bias = self.threshold_bias[index].to(adj.device)
        # 用adj模板+偏置做软门控分数
        gate_input = (mask_template + bias) / temp
        soft_gate = torch.sigmoid(gate_input)
        # 加小epsilon避免完全零
        soft_gate = soft_gate + 1e-7
        masked = soft_gate * adj
        return masked

    def forward(self, adj):
        result = [self._soft_mask(i, a)
                  for i, a in enumerate(adj)]
        if _FX_DBG:
            temp = torch.abs(self.temperature).item()
            gate0 = torch.sigmoid(
                (self.mask[0] +
                 self.threshold_bias[0]) / temp)
            print(f"[FX:mask] temperature={temp:.4f} "
                  f"gate_mean={gate0.mean().item():.4f} "
                  f"gate_sparsity="
                  f"{(gate0 < 0.1).float().mean().item():.2%}",
                  file=sys.stderr)
        return result
