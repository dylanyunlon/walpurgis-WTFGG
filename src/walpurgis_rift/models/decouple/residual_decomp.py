"""Rift ResidualDecomp — from upstream with debug hooks"""
import torch.nn as nn
import sys, os

_RF_DBG = os.environ.get('RIFT_DEBUG', '0') == '1'


class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        if _RF_DBG:
            print(f"[RF-DBG:residual_decomp] residual_norm={u.norm().item():.4f}",
                  file=sys.stderr, flush=True)
        return u
