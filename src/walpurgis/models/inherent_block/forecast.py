"""
Walpurgis v2 Inherent Forecast — with Gradient Checkpointing
===============================================================
Delta: wraps FC layers in torch.utils.checkpoint for memory savings
during backward pass in deep decouple stacks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Forecast(nn.Module):
    _n = 0

    def __init__(self, hidden_dim, forecast_dim, output_seq_len=12, gap=1):
        super().__init__()
        self.fc_in = nn.Linear(hidden_dim, forecast_dim)
        self.fc_out = nn.Linear(forecast_dim, output_seq_len // gap)
        self.drop = 0.1
        self._debug = True

    def _inner(self, h):
        """Separated for gradient checkpointing."""
        h = F.relu(self.fc_in(h))
        if self.training:
            h = F.dropout(h, p=self.drop)
        return h

    def forward(self, hidden):
        Forecast._n += 1
        h = hidden.mean(dim=1)

        # Gradient checkpoint to save memory in deep stacks
        if self.training and h.requires_grad:
            h = checkpoint(self._inner, h, use_reentrant=False)
        else:
            h = self._inner(h)

        if self._debug and Forecast._n % 1000 == 1:
            print(f"        [InhForecast #{Forecast._n}] h_norm={h.norm().item():.4f}")
        return h
