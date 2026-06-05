"""
residual_decomp.py — CardGame ResidualDecomp
算法改写 (vs upstream):
  - ReLU → CELU (连续可微, 负半轴也有梯度)
  - 新增可学习残差缩放参数β: u = x - β*CELU(y)
  - β初始化为1.0, 模型可学习最优残差缩放比例
"""
import os
import sys
import torch
import torch.nn as nn

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="ResDecomp"):
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


class ResidualDecomp(nn.Module):
    """CardGame残差分解: CELU激活 + 可学习残差缩放β

    改写:
      - ReLU → CELU(alpha=1.0): 负半轴连续可微, 梯度不为零
      - 引入可学习参数β (初始1.0): residual = x - β * CELU(y)
      - 残差经LayerNorm归一化
    """

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # CELU替代ReLU
        self.ac = nn.CELU(alpha=1.0)
        # 可学习残差缩放 (初始化1.0)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        _dbg("input.x", x)
        _dbg("input.y", y)

        activated = self.ac(y)
        _dbg("celu_output", activated)

        # 可学习缩放beta
        beta_clamped = torch.clamp(self.beta, min=0.01, max=5.0)
        _dbg("beta", beta_clamped)

        u = x - beta_clamped * activated
        u = self.ln(u)
        _dbg("residual_output", u)
        return u
