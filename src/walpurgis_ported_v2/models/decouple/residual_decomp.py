"""
Residual decomposition: strips the learnable component from the signal,
leaving behind only the unexplained residual.
"""

import torch.nn as nn
import sys

_DBG_RESID = ("--debug-resid" in sys.argv) or False


class ResidualDecomp(nn.Module):
    """u = LayerNorm(x − ReLU(y))"""

    def __init__(self, norm_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape[-1])
        self.relu = nn.ReLU()

    def forward(self, x, y):
        residual = x - self.relu(y)
        out = self.layer_norm(residual)
        if _DBG_RESID:
            print(f"[DBG:resid] ResidualDecomp  x_norm={x.norm().item():.4f}  "
                  f"y_norm={y.norm().item():.4f}  residual_norm={out.norm().item():.4f}")
        return out
