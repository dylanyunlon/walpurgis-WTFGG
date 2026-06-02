"""
Walpurgis v2 Residual Decomposition — Affine Scale+Shift with ELU Gate
=========================================================================
Delta vs prior:
  - sigmoid(scalar_scale) → *ELU-gated* per-channel affine:
      g = 1 + elu(linear(backcast_mean))      # per-channel, ≥0
      residual = LN(input − g · backcast − shift)
    This allows the scale to exceed 1 (unlike sigmoid) while remaining
    non-negative via the ELU+1 trick — the network can *amplify* backcast
    removal when the temporal path dominates.
  - Shift is now per-channel instead of scalar.
  - Verbose diagnostics: tracks scale distribution and residual-to-input
    energy ratio for gradient flow monitoring.

Breakpoint helpers:
    self._diag_last         # dict of last forward's statistics
    self.scale_histogram()  # print scale gate distribution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ResidualDecomp(nn.Module):
    _n = 0

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # Per-channel affine gate replaces scalar sigmoid
        self._gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self._gate_proj.weight)
        nn.init.zeros_(self._gate_proj.bias)
        self._shift = nn.Parameter(torch.zeros(dim))
        self._debug = True
        self._diag_last = {}
        self._gate_samples = deque(maxlen=200)

    def _mish_gate(self, backcast):
        """Mish+1 gate (v4). upstream: relu+LN. v3: ELU+1. v4: Mish+1."""
        summary = backcast.mean(dim=tuple(range(backcast.dim() - 1)))
        raw = self._gate_proj(summary)
        return 1.0 + F.mish(raw)

    def scale_histogram(self, bins=10):
        """Print gate value distribution — call from pdb."""
        if not self._gate_samples:
            print("  [ResDecomp] no gate samples yet")
            return
        import numpy as _np
        vals = _np.array(list(self._gate_samples))
        lo, hi = vals.min(), vals.max()
        print(f"  [ResDecomp] gate histogram over {len(vals)} calls:")
        print(f"    range=[{lo:.4f}, {hi:.4f}]  μ={vals.mean():.4f}  σ={vals.std():.4f}")
        counts, edges = _np.histogram(vals, bins=bins)
        for i, c in enumerate(counts):
            bar = "█" * int(c / max(counts) * 30) if max(counts) > 0 else ""
            print(f"    [{edges[i]:.3f},{edges[i+1]:.3f}): {c:4d} {bar}")

    def forward(self, inp, backcast):
        ResidualDecomp._n += 1
        gate = self._mish_gate(backcast)
        # Expand gate to match spatial dims
        shape = [1] * (backcast.dim() - 1) + [gate.shape[-1]]
        gate_exp = gate.view(*shape)

        residual = self.ln(inp - gate_exp * backcast - self._shift)

        # Diagnostics
        if self._debug:
            with torch.no_grad():
                g_mean = gate.mean().item()
                self._gate_samples.append(g_mean)
                inp_en = inp.norm().item()
                res_en = residual.norm().item()
                ratio = res_en / (inp_en + 1e-12)
                self._diag_last = {
                    "step": ResidualDecomp._n,
                    "gate_mean": round(g_mean, 5),
                    "gate_std": round(gate.std().item(), 5),
                    "shift_norm": round(self._shift.norm().item(), 6),
                    "energy_ratio": round(ratio, 4),
                }
            if ResidualDecomp._n % 500 == 1:
                d = self._diag_last
                print(
                    f"      [ResDecomp #{ResidualDecomp._n}] "
                    f"gate: μ={d['gate_mean']:.4f} σ={d['gate_std']:.4f} | "
                    f"shift_‖={d['shift_norm']:.5f} | "
                    f"energy_ratio={d['energy_ratio']:.4f}"
                )
        return residual
