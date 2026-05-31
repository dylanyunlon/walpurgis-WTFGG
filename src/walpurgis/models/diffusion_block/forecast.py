"""
Walpurgis Forecast Head — Temporal Projection with Dropout Scheduling
=======================================================================
Derived from D2STGNN forecast.py.

Change: applies spatial dropout (drops entire node features) instead of
standard element-wise dropout. This is more appropriate for graph data
where node-level features should be dropped as units.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Forecast(nn.Module):
    """Projects hidden features to forecast horizon.
    
    Architecture: hidden → FC1 → ReLU → SpatialDropout → FC2
    
    Walpurgis: uses F.dropout2d on the [B, N, D] tensor to drop
    entire feature channels per node, rather than random elements.
    This provides stronger regularization for spatial-temporal models.
    """
    
    _call_count = 0
    
    def __init__(self, hidden_dim, forecast_dim, output_seq_len=12, gap=1):
        super().__init__()
        self.fc_in = nn.Linear(hidden_dim, forecast_dim)
        self.fc_out = nn.Linear(forecast_dim, output_seq_len // gap)
        self.output_len = output_seq_len
        self.gap = gap
        self.dropout_rate = 0.1  # lighter than standard 0.3
        self._debug_on = True
    
    def forward(self, hidden):
        """
        Args:
            hidden: [B, L, N, D] — temporal hidden features
        Returns:
            forecast: [B, N, forecast_dim] — aggregated forecast embedding
        """
        Forecast._call_count += 1
        
        # Temporal aggregation: mean pool over sequence length
        h = hidden.mean(dim=1)  # [B, N, D]
        
        h = F.relu(self.fc_in(h))
        
        # Spatial dropout: drop entire feature channels per node
        if self.training:
            h = F.dropout2d(h.unsqueeze(-1), p=self.dropout_rate).squeeze(-1)
        
        forecast = self.fc_out(h)
        
        if self._debug_on and Forecast._call_count % 1000 == 1:
            print(f"        [Forecast #{Forecast._call_count}] "
                  f"in_mean_norm={hidden.norm(dim=-1).mean().item():.4f} "
                  f"out_norm={forecast.norm().item():.4f}")
        
        return h  # return pre-projection hidden for aggregation in model.py
