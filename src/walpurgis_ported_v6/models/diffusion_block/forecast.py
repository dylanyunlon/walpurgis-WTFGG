"""Diffusion forecast branch — adaptive-gap variant.

Changes
-------
The upstream uses a fixed ``gap`` to decide how many auto-regressive
steps to unroll.  This version computes ``effective_gap`` as
``min(gap, remaining_seq)`` at each step, so when ``output_seq_len``
is not evenly divisible by ``gap`` the last step naturally covers
the remainder instead of silently dropping time steps.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.gap = model_args['gap']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = [hidden_states_dif[:, -1, :, :].unsqueeze(1)]
        history = gated_history_data

        # adaptive gap: handle non-divisible seq lengths
        produced = self.gap        # first step already produces `gap` slots
        while produced < self.output_seq_len:
            remaining = self.output_seq_len - produced
            step_size = min(self.gap, remaining)      # ← adaptive

            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                pad = history[:, -(self.k_t - len(recent)):, :, :]
                recent = [pad] + recent
            stacked = torch.cat(recent, dim=1)
            predict.append(localized_st_conv(stacked,
                                             dynamic_graph, static_graph))
            produced += step_size

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)

        _dbg("DifForecast", predict, n_steps=len(predict))
        return predict
