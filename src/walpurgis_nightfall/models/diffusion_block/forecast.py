"""
Forecast (diffusion) — Nightfall变体
算法改写:
  1. forecast_fc前加GELU激活 + LayerNorm
  2. 自回归循环中间加状态打印
"""
import torch
import torch.nn as nn
from walpurgis_nightfall import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        # GELU + LayerNorm + 投影
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        num_ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step_i in range(num_ar_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            ar_out = localized_st_conv(_1, dynamic_graph, static_graph)
            predict.append(ar_out)
            if step_i == 0:
                _dbg("dif_forecast.ar_step0", ar_out, "model")
        predict = torch.cat(predict, dim=1)
        # GELU + LayerNorm + 投影
        predict = self.pre_norm(predict)
        predict = self.act(predict)
        predict = self.forecast_fc(predict)
        _dbg("dif_forecast.output", predict, "model")
        return predict
