"""Cascade diffusion forecast: BatchNorm stabilization + gradient-friendly projection.
Unlike upstream (no norm) and vortex (GroupNorm + linear interpolation padding),
Cascade uses BatchNorm on the forecast projection output for stable training
across varying batch statistics, and zero-padded history fallback with learned bias."""
import torch
import torch.nn as nn
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc    = nn.Linear(hidden_dim, forecast_hidden_dim)
        # Cascade特有: BatchNorm on forecast output for stable gradient flow
        self.bn = nn.BatchNorm1d(forecast_hidden_dim)
        # Learned padding bias when history is insufficient
        self.pad_bias = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.model_args     = model_args

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                # Cascade: add learned padding bias for boundary smoothing
                _2 = _2 + self.pad_bias
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        proj = self.forecast_fc(predict)
        # Cascade特有: BatchNorm on last dim
        B, S, N, C = proj.shape
        proj_flat = proj.reshape(B * S * N, C)
        proj_flat = self.bn(proj_flat)
        proj = proj_flat.reshape(B, S, N, C)
        if _CAS_DBG:
            print(f"[CAS:forecast@dif_forecast] steps={proj.shape[1]} "
                  f"range=[{proj.min().item():.4f},{proj.max().item():.4f}]",
                  file=sys.stderr)
        return proj
