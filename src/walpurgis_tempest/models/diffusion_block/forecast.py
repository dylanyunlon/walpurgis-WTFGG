"""Tempest diffusion forecast: linear interpolation padding + GroupNorm.
Unlike upstream (last-sample repeat, no norm) and eclipse (reflect padding, LayerNorm),
Tempest uses linear interpolation for smoother padding transitions and GroupNorm
for better generalization across different group sizes."""
import torch, torch.nn as nn, sys, os
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']; self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        # GroupNorm (vs upstream none, eclipse LayerNorm)
        n_groups = min(4, forecast_hidden_dim)
        self.norm = nn.GroupNorm(n_groups, forecast_hidden_dim)
        self.model_args = model_args

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = [hidden_states_dif[:, -1, :, :].unsqueeze(1)]
        history = gated_history_data
        for step in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                sub = self.k_t - len(recent)
                # Linear interpolation padding: smoothly extrapolate from history
                hist_end = history[:, -1:, :, :]  # last history
                pred_start = recent[0]  # first prediction
                weights = torch.linspace(0, 1, sub + 2, device=hist_end.device)[1:-1]
                interp_pad = []
                for w in weights:
                    interp_pad.append(hist_end * (1 - w) + pred_start * w)
                recent = interp_pad + recent
            recent = torch.cat(recent, dim=1)
            predict.append(localized_st_conv(recent, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        proj = self.forecast_fc(predict)
        # GroupNorm: need to permute to [B, C, ...] format
        B, S, N, C = proj.shape
        proj = proj.permute(0, 3, 1, 2)  # [B, C, S, N]
        proj = self.norm(proj)
        proj = proj.permute(0, 2, 3, 1)  # [B, S, N, C]
        if _TEM_DBG:
            print(f"[TEM:forecast@dif_forecast] steps={proj.shape[1]} range=[{proj.min().item():.4f},{proj.max().item():.4f}]", file=sys.stderr)
        return proj
