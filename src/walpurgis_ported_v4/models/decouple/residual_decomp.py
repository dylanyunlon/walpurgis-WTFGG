"""
ResidualDecomp — walpurgis_ported_v4
Modifications:
  - Replaced ReLU with LeakyReLU(negative_slope=0.1) to preserve weak
    negative gradient flow during decomposition
  - forward() prints residual magnitude stats
"""
import torch.nn as nn
import sys

_V4_DEBUG = True


class ResidualDecomp(nn.Module):
    """Residual decomposition: u = LayerNorm(x - act(y))
    v4: uses LeakyReLU to allow negative gradient flow.
    """

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.LeakyReLU(negative_slope=0.1)  # v4: changed from ReLU

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)

        if _V4_DEBUG:
            residual_norm = u.detach().norm().item()
            x_norm = x.detach().norm().item()
            ratio = residual_norm / (x_norm + 1e-8)
            print(f"[v4-DBG][ResidualDecomp] "
                  f"||residual||={residual_norm:.4f} ||x||={x_norm:.4f} "
                  f"ratio={ratio:.4f} shape={tuple(u.shape)}",
                  file=sys.stderr)
        return u
