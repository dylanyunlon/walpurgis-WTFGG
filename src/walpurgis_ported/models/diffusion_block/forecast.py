"""
Walpurgis v2 Diffusion Forecast Head — Squeeze-Excite Regularization
========================================================================
Delta vs prior:
  - Channel-shuffle dropout → *squeeze-and-excite* (SE) attention before
    dropout.  SE produces a per-channel importance weight via global
    average pool → FC → sigmoid, which acts as learned feature selection
    complementary to dropout's random selection.
  - SE ratio is r=4 (reduces channels by 4x in the bottleneck).
  - Tracks output norm ratio for gradient flow diagnostics.

Breakpoint helpers:
    self._diag_last    # dict with last forward stats
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Forecast(nn.Module):
    _n = 0

    def __init__(self, hidden_dim, forecast_dim, output_seq_len=12, gap=1):
        super().__init__()
        self.fc_in = nn.Linear(hidden_dim, forecast_dim)
        self.fc_out = nn.Linear(forecast_dim, output_seq_len // gap)
        self.drop_rate = 0.1
        # Squeeze-and-Excite channel attention
        se_mid = max(forecast_dim // 4, 8)
        self._se_fc1 = nn.Linear(forecast_dim, se_mid)
        self._se_fc2 = nn.Linear(se_mid, forecast_dim)
        self._debug = True
        self._diag_last = {}

    def _squeeze_excite(self, h):
        """CBAM-lite (v4): avg+max pool. upstream: none. v3: SE."""
        if h.dim() >= 3 and h.shape[1] > 1:
            avg_p, max_p = h.mean(dim=1), h.max(dim=1).values
        else:
            avg_p = h.squeeze(1) if h.dim()==3 else h
            max_p = avg_p
        gate = torch.sigmoid(
            self._se_fc2(F.relu(self._se_fc1(avg_p))) +
            self._se_fc2(F.relu(self._se_fc1(max_p)))
        )
        # Expand and apply
        if h.dim() == 3:
            gate = gate.unsqueeze(1)
        return h * gate

    def forward(self, hidden):
        Forecast._n += 1
        h = hidden.mean(dim=1)
        in_norm = h.norm().item() if self._debug else 0

        h = F.relu(self.fc_in(h))

        if self.training:
            h = self._squeeze_excite(h)
            h = F.dropout(h, p=self.drop_rate)

        if self._debug:
            with torch.no_grad():
                out_norm = h.norm().item()
                self._diag_last = {
                    "step": Forecast._n,
                    "in_norm": round(in_norm, 4),
                    "out_norm": round(out_norm, 4),
                    "ratio": round(out_norm / (in_norm + 1e-8), 4),
                }
            if Forecast._n % 1000 == 1:
                d = self._diag_last
                print(
                    f"        [DifForecast #{Forecast._n}] "
                    f"in_‖={d['in_norm']:.4f} out_‖={d['out_norm']:.4f} "
                    f"ratio={d['ratio']:.4f} | SE active"
                )
        return h
