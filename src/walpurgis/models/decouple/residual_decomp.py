"""
Walpurgis v2 Residual Decomposition — Scale + Shift
=====================================================
Delta: single learnable scale → *scale+shift* pair.
residual = LN(input − sigmoid(s)·backcast − shift)
The shift absorbs systematic bias in the backcast prediction.
"""
import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    _n = 0

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self._scale = nn.Parameter(torch.tensor(0.5))
        self._shift = nn.Parameter(torch.tensor(0.0))
        self._debug = True

    def forward(self, inp, backcast):
        ResidualDecomp._n += 1
        s = torch.sigmoid(self._scale)
        residual = self.ln(inp - s * backcast - self._shift)

        if self._debug and ResidualDecomp._n % 500 == 1:
            with torch.no_grad():
                ratio = backcast.norm().item() / (inp.norm().item() + 1e-8)
                print(
                    f"      [ResDecomp #{ResidualDecomp._n}] "
                    f"scale={s.item():.4f} shift={self._shift.item():.5f} "
                    f"back/inp={ratio:.4f}"
                )
        return residual
