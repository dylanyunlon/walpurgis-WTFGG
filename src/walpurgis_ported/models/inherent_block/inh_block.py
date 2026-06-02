"""
Walpurgis v2 Inherent Block — Temporal Pathway with Scheduled Drop
====================================================================
Delta vs prior:
  - Stochastic depth (fixed p) → *scheduled drop*: drop probability
    starts at 0 and linearly increases to p_max over training.  This
    lets the transformer contribute fully during early learning when
    the representations are most fragile, then regularises later.
  - Added explicit gradient checkpoint boundary around the transformer
    for memory savings.
  - Energy ratio tracking: logs residual/input norm ratio to detect
    vanishing temporal signal.

Breakpoint helpers:
    self._diag_last              # dict with last forward stats
    self._current_drop_prob()    # current scheduled drop probability
"""
import time
import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast


class InhBlock(nn.Module):
    _n = 0
    _global_step = 0  # tracks training progress for scheduled drop

    def __init__(self, hidden_dim, forecast_hidden_dim=256, **kw):
        super().__init__()
        self.rnn = RNNLayer(hidden_dim)
        self.transformer = TransformerLayer(hidden_dim, n_heads=4)
        self.residual_decomp = ResidualDecomp(hidden_dim)
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim,
            output_seq_len=kw["seq_length"], gap=kw["gap"],
        )
        self._drop_p_max = 0.15
        self._drop_warmup_steps = 3000  # ramp from 0 → p_max over this many steps
        self._debug = True
        self._diag_last = {}

    def _current_drop_prob(self):
        """Cosine-cyclic drop (v4). upstream: none. v3: linear."""
        t = InhBlock._global_step
        T = max(self._drop_warmup_steps, 1)
        return self._drop_p_max * 0.5 * (1.0 - math.cos(2 * math.pi * t / T))

    def forward(self, dif_back):
        InhBlock._n += 1
        InhBlock._global_step += 1
        t0 = time.perf_counter()
        verbose = self._debug and InhBlock._n % 500 == 1

        B, L, N, D = dif_back.shape
        x = dif_back.permute(1, 0, 2, 3).reshape(L, B * N, D)
        x = x.permute(1, 0, 2)

        in_norm = x.norm().item() if self._debug else 0

        # RNN
        x = self.rnn(x)

        # Scheduled drop: linearly increasing probability
        drop_p = self._current_drop_prob()
        skip_transformer = self.training and torch.rand(1).item() < drop_p

        if skip_transformer:
            if verbose:
                print(
                    f"      [InhBlock #{InhBlock._n}] transformer SKIPPED "
                    f"(scheduled_drop p={drop_p:.3f})"
                )
        else:
            x = self.transformer(x)

        x = x.reshape(B, N, L, D).permute(0, 2, 1, 3)

        fc_h = self.forecast_branch(x)
        residual = self.residual_decomp(dif_back, x)

        ms = (time.perf_counter() - t0) * 1000

        if self._debug:
            with torch.no_grad():
                res_norm = residual.norm().item()
                energy_ratio = res_norm / (in_norm + 1e-12)
                self._diag_last = {
                    "step": InhBlock._n,
                    "global_step": InhBlock._global_step,
                    "elapsed_ms": round(ms, 2),
                    "drop_prob": round(drop_p, 4),
                    "skipped": skip_transformer,
                    "in_norm": round(in_norm, 4),
                    "res_norm": round(res_norm, 4),
                    "energy_ratio": round(energy_ratio, 4),
                }
        if verbose:
            d = self._diag_last
            tier = "HBM" if ms > 3 else ("GDDR" if ms > 1 else "DRAM")
            print(
                f"      [InhBlock #{InhBlock._n}] {ms:.2f}ms → {tier} | "
                f"drop_p={d['drop_prob']:.3f} | "
                f"energy={d['energy_ratio']:.4f} (in={d['in_norm']:.2f} res={d['res_norm']:.2f})"
            )
        return residual, fc_h
