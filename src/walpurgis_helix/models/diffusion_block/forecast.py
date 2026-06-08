import torch
import torch.nn as nn

class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc    = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args     = model_args
        # Helix特有: forecast门控 — 可学习的输出缩放
        self.forecast_gate = nn.Parameter(torch.ones(forecast_hidden_dim))

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        # Helix特有: 对forecast输出做sigmoid门控缩放
        gate = torch.sigmoid(self.forecast_gate)
        predict = predict * gate
        return predict
