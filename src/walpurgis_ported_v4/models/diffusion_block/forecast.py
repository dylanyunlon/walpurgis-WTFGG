"""
Forecast (diffusion) — walpurgis_ported_v4
Modifications:
  - AR loop: added step counter debug print showing predicted-vs-remaining
  - forecast_fc: added GELU activation before projection (original: raw linear)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

_V4_DEBUG = True


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step in range(n_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            pred_step = localized_st_conv(_1, dynamic_graph, static_graph)
            predict.append(pred_step)

            if _V4_DEBUG and step < 3:
                print(f"[v4-DBG][DifForecast] AR step {step+1}/{n_steps} "
                      f"pred_shape={tuple(pred_step.shape)} "
                      f"norm={pred_step.detach().norm().item():.4f}",
                      file=sys.stderr)

        predict = torch.cat(predict, dim=1)
        # v4: GELU before projection for smoother gradient landscape
        predict = self.forecast_fc(F.gelu(predict))
        return predict
