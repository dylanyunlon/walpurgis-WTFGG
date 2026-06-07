"""Eclipse inherent forecast: step-decay weighting."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']; self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.step_decay = 0.05  # exponential decay per step

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, F = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        for step in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            _gru = rnn_layer.ln(_gru.squeeze(0)).unsqueeze(0)  # LN after GRU
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None: RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            # Step-decay weighting: farther predictions weighted less
            weight = torch.exp(torch.tensor(-self.step_decay * (step + 1)))
            predict.append(_Z * weight)
        predict = torch.cat(predict, dim=0).reshape(-1, B, N, F).transpose(0, 1)
        predict = self.forecast_fc(predict)
        if _ECL_DBG: print(f"[ECL:inh_forecast] steps={predict.shape[1]} range=[{predict.min().item():.4f},{predict.max().item():.4f}]", file=sys.stderr)
        return predict
