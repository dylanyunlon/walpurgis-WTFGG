"""Eclipse diffusion forecast: reflect padding + LayerNorm."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']; self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.norm = nn.LayerNorm(forecast_hidden_dim)
        self.model_args = model_args

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = [hidden_states_dif[:, -1, :, :].unsqueeze(1)]
        history = gated_history_data
        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                sub = self.k_t - len(recent)
                # Reflect padding instead of last-sample repeat
                pad = history[:, -sub:, :, :].flip(1)
                recent = [pad] + recent
            recent = torch.cat(recent, dim=1)
            predict.append(localized_st_conv(recent, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.norm(self.forecast_fc(predict))
        if _ECL_DBG: print(f"[ECL:dif_forecast] steps={predict.shape[1]} range=[{predict.min().item():.4f},{predict.max().item():.4f}]", file=sys.stderr)
        return predict
