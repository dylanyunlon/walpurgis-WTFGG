"""Nebula inherent forecast: IndRNN + Fourier PE + flash attention autoregression."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes, num_feat] = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            # IndRNN step
            _rnn = rnn_layer.ind_rnn_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _rnn], dim=0)
            # Fourier PE
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Flash attention
            _Z = transformer_layer(_rnn, K_in=RNN_H, V_in=RNN_H)
            predict.append(_Z)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        if _NEB_DBG:
            print(f"[NEB:forecast@inh_forecast] shape={list(predict.shape)}", file=sys.stderr)
        return predict
