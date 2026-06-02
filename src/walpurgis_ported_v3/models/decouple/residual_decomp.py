"""
Residual decomposition: subtracts the learned component and normalizes.
"""
import sys
import torch.nn as nn

_DBG = ("--debug-resdecomp" in sys.argv)


class ResidualDecomp(nn.Module):

    def __init__(self, norm_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape[-1])
        self.relu = nn.ReLU()

    def forward(self, x, y):
        residual = x - self.relu(y)
        out = self.layer_norm(residual)
        if _DBG:
            print(f"[DBG:resdecomp] in x={tuple(x.shape)}  "
                  f"y_mean={y.mean().item():.4f}  "
                  f"out_std={out.std().item():.4f}")
        return out
