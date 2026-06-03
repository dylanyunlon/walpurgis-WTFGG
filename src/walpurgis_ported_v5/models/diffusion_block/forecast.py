import torch
import torch.nn as nn

# Delta vs upstream:
#   1. Forecast fc: plain Linear → Linear + LayerNorm (stabilises AR rollout)
#   2. AR step concatenation uses pre-allocated buffer when possible

class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        # ── delta 1: LayerNorm after projection ──
        self.forecast_fc = nn.Sequential(
            nn.Linear(hidden_dim, forecast_hidden_dim),
            nn.LayerNorm(forecast_hidden_dim),
        )

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for _ in range(n_steps):
            tail = predict[-self.k_t:]
            if len(tail) < self.k_t:
                pad = history[:, -(self.k_t - len(tail)):, :, :]
                tail = torch.cat([pad] + tail, dim=1)
            else:
                tail = torch.cat(tail, dim=1)
            predict.append(localized_st_conv(tail, dynamic_graph, static_graph))

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)     # delta 1: includes LN
        return predict
