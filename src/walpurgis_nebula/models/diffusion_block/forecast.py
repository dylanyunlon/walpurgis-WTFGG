"""Nebula diffusion forecast: GroupNorm + residual gating."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        # Nebula: residual gating for forecast steps
        self.step_gate = nn.Linear(hidden_dim, hidden_dim)
        self.model_args = model_args

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            step_out = localized_st_conv(_1, dynamic_graph, static_graph)
            # Nebula: sigmoid gated residual for each forecast step
            gate = torch.sigmoid(self.step_gate(step_out))
            step_out = gate * step_out + (1.0 - gate) * predict[-1]
            predict.append(step_out)
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        if _NEB_DBG:
            print(f"[NEB:forecast@dif_forecast] shape={list(predict.shape)} norm={predict.norm().item():.4f}", file=sys.stderr)
        return predict
