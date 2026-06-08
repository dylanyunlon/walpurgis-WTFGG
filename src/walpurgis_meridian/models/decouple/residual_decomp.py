"""Meridian ResidualDecomp — gated residual with learnable interpolation.
Changes vs upstream:
  - Learnable gate alpha for soft interpolation: out = alpha*LN(x-y) + (1-alpha)*x
  - Uses GELU instead of ReLU on the extracted component
  - Debug: prints interpolation alpha and norms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # learnable interpolation between full residual and soft residual
        self.gate_alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, y):
        extracted = F.gelu(y)
        residual = self.ln(x - extracted)
        alpha = torch.sigmoid(self.gate_alpha)
        out = alpha * residual + (1.0 - alpha) * x
        if _DBG:
            print(f"[MER:res_decomp] alpha={alpha.item():.4f} "
                  f"residual_norm={residual.detach().norm().item():.4f} "
                  f"x_norm={x.detach().norm().item():.4f}",
                  file=sys.stderr)
        return out
