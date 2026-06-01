"""
Walpurgis v2 Diffusion Forecast Head
=======================================
Delta: spatial dropout → *channel-shuffle then standard dropout*.
Channel shuffle provides inter-node feature mixing before dropout,
acting as implicit regularisation.
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
        self._debug = True

    def _channel_shuffle(self, x, groups=4):
        """Shuffle feature channels across groups for inter-node mixing."""
        B, N, C = x.shape
        if C % groups != 0:
            return x
        x = x.view(B, N, groups, C // groups)
        x = x.transpose(2, 3).contiguous()
        return x.view(B, N, C)

    def forward(self, hidden):
        Forecast._n += 1
        h = hidden.mean(dim=1)
        h = F.relu(self.fc_in(h))

        if self.training:
            h = self._channel_shuffle(h)
            h = F.dropout(h, p=self.drop_rate)

        if self._debug and Forecast._n % 1000 == 1:
            print(
                f"        [DifForecast #{Forecast._n}] "
                f"in_norm={hidden.norm(dim=-1).mean().item():.4f} "
                f"out_norm={h.norm().item():.4f}"
            )
        return h
