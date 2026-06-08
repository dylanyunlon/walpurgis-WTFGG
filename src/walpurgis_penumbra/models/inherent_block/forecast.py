"""
Forecast (inherent branch) — Penumbra变体
适配MinGRU和CrossAttention接口
"""
import torch
import torch.nn as nn
from ... import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer,
                rnn_layer, pe):
        batch_size, _, num_nodes, num_feat = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(
                int(self.output_seq_len
                    / self.model_args['gap']) - 1):
            # MinGRU一步: 手动update
            prev = predict[-1][0]  # [B*N, D]
            z = torch.sigmoid(
                rnn_layer.W_z(
                    torch.cat([RNN_H[-1], prev], dim=-1)))
            h_tilde = torch.tanh(rnn_layer.W_h(prev))
            _gru = ((1 - z) * RNN_H[-1]
                    + z * h_tilde).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Cross-attention一步
            _Z = transformer_layer(
                _gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("inh_forecast.output",
             predict, "inherent")
        return predict
