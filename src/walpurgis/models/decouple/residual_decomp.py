import time

import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """Residual decomposition: u = LayerNorm(x - ReLU(y)).

    Extracts the residual component after removing the predicted
    (backcast) signal, then normalizes for stable gradient flow.

    Walpurgis notes:
    - This is a lightweight op (elementwise sub + LN); DRAM tier is fine.
    - Residual magnitude relative to input is tracked — if the residual
      is consistently near-zero, the backcast branch has collapsed to
      identity (degenerate).
    """

    _call_count = 0

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()
        print(f"[Walpurgis::ResidualDecomp] init norm_dim={input_shape[-1]}")

    def forward(self, x, y):
        ResidualDecomp._call_count += 1
        _verbose = (ResidualDecomp._call_count <= 5 or ResidualDecomp._call_count % 500 == 0)

        t0 = time.perf_counter()
        u = x - self.ac(y)
        u = self.ln(u)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            residual_ratio = u.abs().mean().item() / (x.abs().mean().item() + 1e-8)
            print(f"[Walpurgis::ResidualDecomp::forward] call#{ResidualDecomp._call_count} "
                  f"elapsed={elapsed_ms:.3f}ms "
                  f"|residual|/|input|={residual_ratio:.4f} "
                  f"output mean={u.mean().item():.6f} std={u.std().item():.6f}")
            if residual_ratio < 0.01:
                print(f"  ⚠ ResidualDecomp: residual near zero — backcast may have collapsed")

        return u
