"""
Walpurgis Inherent Forecast — Temporal Projection
===================================================
Same structure as diffusion forecast but processes inherent (temporal) features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Forecast(nn.Module):
    """Temporal forecast head: pool + project."""
    
    _call_count = 0
    
    def __init__(self, hidden_dim, forecast_dim, output_seq_len=12, gap=1):
        super().__init__()
        self.fc_in = nn.Linear(hidden_dim, forecast_dim)
        self.fc_out = nn.Linear(forecast_dim, output_seq_len // gap)
        self.dropout_rate = 0.1
        self._debug_on = True
    
    def forward(self, hidden):
        Forecast._call_count += 1
        
        h = hidden.mean(dim=1)
        h = F.relu(self.fc_in(h))
        
        if self.training:
            h = F.dropout(h, p=self.dropout_rate)
        
        if self._debug_on and Forecast._call_count % 1000 == 1:
            print(f"        [InhForecast #{Forecast._call_count}] "
                  f"h_norm={h.norm().item():.4f}")
        
        return h
