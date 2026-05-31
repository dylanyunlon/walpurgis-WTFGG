"""
Walpurgis v2 Inherent Block — Temporal Pathway with Stochastic Depth
=====================================================================
Delta: learnable PE removed (RoPE is in the Transformer now); added
*stochastic depth* — randomly skips the transformer branch with
probability p_drop during training.  This regularises the temporal
path and prevents over-reliance on the transformer.
"""
import time
import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast


class InhBlock(nn.Module):
    _n = 0

    def __init__(self, hidden_dim, forecast_hidden_dim=256, **kw):
        super().__init__()
        self.rnn = RNNLayer(hidden_dim)
        self.transformer = TransformerLayer(hidden_dim, n_heads=4)
        self.residual_decomp = ResidualDecomp(hidden_dim)
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim,
            output_seq_len=kw["seq_length"], gap=kw["gap"],
        )
        self._stoch_depth_p = 0.1   # drop probability for transformer
        self._debug = True

    def forward(self, dif_back):
        InhBlock._n += 1
        t0 = time.perf_counter()
        verbose = self._debug and InhBlock._n % 500 == 1

        B, L, N, D = dif_back.shape
        x = dif_back.permute(1, 0, 2, 3).reshape(L, B * N, D)
        x = x.permute(1, 0, 2)  # [B*N, L, D]

        # RNN
        x = self.rnn(x)

        # Stochastic depth: skip transformer with prob p during training
        if self.training and torch.rand(1).item() < self._stoch_depth_p:
            if verbose:
                print(f"      [InhBlock #{InhBlock._n}] transformer SKIPPED (stoch_depth)")
        else:
            x = self.transformer(x)

        x = x.reshape(B, N, L, D).permute(0, 2, 1, 3)

        fc_h = self.forecast_branch(x)
        residual = self.residual_decomp(dif_back, x)

        ms = (time.perf_counter() - t0) * 1000
        if verbose:
            tier = "HBM" if ms > 3 else ("GDDR" if ms > 1 else "DRAM")
            print(
                f"      [InhBlock #{InhBlock._n}] {ms:.2f}ms → {tier} | "
                f"residual_norm={residual.norm().item():.4f}"
            )
        return residual, fc_h
