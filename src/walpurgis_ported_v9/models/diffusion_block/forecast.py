"""
forecast.py (diffusion) — v9 port
Algo delta:
  1. AR 展开每步之间加 Dropout(0.1), 抑制自回归误差累积
  2. gap 不整除 output_seq_len 时用 ceil 多预测一步再截断,
     upstream 直接 int() 截断会丢最后一段
  3. 最终 forecast_fc 前加 LayerNorm
"""
import math
import torch
import torch.nn as nn
from walpurgis_ported_v9 import _dbg

_TAG = "dif_fc"


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.gap = model_args['gap']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        # v9
        self.ar_dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = [hidden_states_dif[:, -1, :, :].unsqueeze(1)]

        # v9: ceil instead of int truncation
        n_steps = math.ceil(self.output_seq_len / self.gap) - 1

        history = gated_history_data
        for step in range(n_steps):
            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                pad = history[:, -(self.k_t - len(recent)):, :, :]
                recent = [pad] + recent
            cat = torch.cat(recent, dim=1)
            # v9: dropout between AR steps
            cat = self.ar_dropout(cat)
            nxt = localized_st_conv(cat, dynamic_graph, static_graph)
            predict.append(nxt)

        predict = torch.cat(predict, dim=1)
        # v9: LN before FC
        predict = self.ln(predict)
        predict = self.forecast_fc(predict)

        _dbg(_TAG, f"dif_forecast  steps={n_steps+1}  out_shape={list(predict.shape)}")
        return predict
