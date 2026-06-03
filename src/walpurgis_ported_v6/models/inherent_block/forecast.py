"""Inherent forecast branch — with decaying positional confidence.

Changes
-------
Each auto-regressive forecast step is multiplied by a decay factor
``gamma^step`` (gamma=0.95 by default, learnable).  The intuition:
later time steps are less reliable because they compound GRU/Transformer
approximation error.  The decay lets the output layer implicitly
down-weight far-future predictions during the summation with the
diffusion forecast branch.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.gap = model_args['gap']
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # learnable step-decay
        self.log_gamma = nn.Parameter(torch.tensor(-0.05))   # ≈ 0.95

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        gamma = torch.sigmoid(self.log_gamma + 0.5)  # keep in (0,1)

        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = int(self.output_seq_len / self.gap) - 1

        for step in range(n_steps):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1])
            _gru = rnn_layer.ln(_gru)    # use the LN from RNNLayer
            _gru = _gru.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            # apply step-wise decay
            decay = gamma ** (step + 1)
            _Z = _Z * decay
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("InhForecast", predict, gamma=gamma.item())
        return predict
