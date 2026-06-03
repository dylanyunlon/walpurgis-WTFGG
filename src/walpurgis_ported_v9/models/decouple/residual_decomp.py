"""
residual_decomp.py — v9 port
Algo delta:
  1. ReLU → SiLU (x·sigmoid(x)), 保留负值信息
  2. 可学习 scale 参数 γ (init=1.0): u = x − γ·SiLU(y)
     让网络自己决定残差分解的力度
  3. LayerNorm 可选 pre-norm (先 LN 再减) vs post-norm (先减再 LN),
     默认 post-norm 与 upstream 行为一致, 但 pre_ln=True 时切换
"""
import torch
import torch.nn as nn
from walpurgis_ported_v9 import _dbg

_TAG = "res_decomp"


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape, pre_ln: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.act = nn.SiLU()                            # v9: SiLU
        self.scale = nn.Parameter(torch.ones(1))        # v9: learnable scale
        self.pre_ln = pre_ln

    def forward(self, x, y):
        scale_val = torch.clamp(self.scale, min=0.01)
        if self.pre_ln:
            x = self.ln(x)
            u = x - scale_val * self.act(y)
        else:
            u = x - scale_val * self.act(y)
            u = self.ln(u)

        _dbg(_TAG, f"scale={scale_val.item():.4f}  "
                    f"|u|_mean={u.abs().mean().item():.6g}  pre_ln={self.pre_ln}")
        return u
