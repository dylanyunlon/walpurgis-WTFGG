"""
Walpurgis Residual Decomposition — Adaptive Gated Residual
============================================================
Adapted from D2STGNN ResidualDecomp.

Algorithm change:
  D2STGNN: u = LayerNorm(x - ReLU(y))
  Walpurgis: u = LayerNorm(alpha * x + (1 - alpha) * (x - ReLU(y)))
  where alpha is a learnable scalar initialized to 0.1.

  This "gated residual" lets the model learn how much of the original
  signal to preserve vs how much to decompose. In early training,
  alpha ≈ 0.1 means mostly decomposition (close to D2STGNN behavior).
  As training progresses, if the decomposition is noisy, the model can
  increase alpha to preserve more of the original signal.
"""

import time
import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """Gated residual decomposition with adaptive mixing.

    The mixing parameter alpha is logged for diagnostics — if it drifts
    toward 1.0, it means the backcast/forecast branches are not providing
    useful decomposition and may need debugging.
    """

    _call_count = 0

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

        # Walpurgis: learnable residual gate
        # Initialized to 0.1 (mostly decomposition, close to D2STGNN)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._alpha_history = []

    def forward(self, x, y):
        ResidualDecomp._call_count += 1
        _verbose = (ResidualDecomp._call_count <= 5 or
                    ResidualDecomp._call_count % 500 == 0)

        # D2STGNN: u = x - ReLU(y)
        # Walpurgis: gated mix of identity and decomposition
        alpha = torch.sigmoid(self.alpha)  # constrain to (0, 1)
        decomposed = x - self.ac(y)
        u = alpha * x + (1.0 - alpha) * decomposed
        u = self.ln(u)

        self._alpha_history.append(alpha.item())

        if _verbose:
            decomp_norm = decomposed.norm().item()
            out_norm = u.norm().item()
            x_norm = x.norm().item()
            print(f"    [ResidualDecomp] call#{ResidualDecomp._call_count} "
                  f"alpha={alpha.item():.4f} x_norm={x_norm:.4f} "
                  f"decomp_norm={decomp_norm:.4f} out_norm={out_norm:.4f}")
            if alpha.item() > 0.8:
                print(f"    ⚠ alpha={alpha.item():.3f} — model is bypassing "
                      f"decomposition (preserving original signal)")

        return u
