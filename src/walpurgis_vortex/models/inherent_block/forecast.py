"""Vortex inherent forecast: cosine annealing weighting for future steps.
Unlike upstream (uniform weight) and eclipse (exponential step decay),
Vortex weights future predictions with a cosine schedule, providing
a smooth warm-to-cold transition that is less aggressive than exponential."""
import math, torch, torch.nn as nn, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']; self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, F = X.shape
        total_steps = int(self.output_seq_len / self.model_args['gap'])
        predict = [Z[-1, :, :].unsqueeze(0)]
        for step in range(total_steps - 1):
            _gru = rnn_layer.mingru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            # GroupNorm after MinGRU
            _gru_normed = rnn_layer.norm(_gru.squeeze(0).unsqueeze(-1)).squeeze(-1).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru_normed], dim=0)
            if pe is not None: RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru_normed, K=RNN_H, V=RNN_H)
            # Cosine annealing weight: w(t) = 0.5*(1 + cos(pi*t/T))
            progress = (step + 1) / max(total_steps - 1, 1)
            weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            predict.append(_Z * max(weight, 0.1))  # floor at 0.1 to avoid vanishing
        predict = torch.cat(predict, dim=0).reshape(-1, B, N, F).transpose(0, 1)
        predict = self.forecast_fc(predict)
        if _VX_DBG:
            print(f"[VX:forecast@inh_forecast] steps={predict.shape[1]} range=[{predict.min().item():.4f},{predict.max().item():.4f}]", file=sys.stderr)
        return predict
