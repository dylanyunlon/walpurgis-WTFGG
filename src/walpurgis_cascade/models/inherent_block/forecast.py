"""Cascade inherent forecast: step-weighted projection with learnable horizon emphasis.
Unlike upstream (uniform weight) and vortex (cosine annealing weighting),
Cascade uses a learnable per-step weight vector that allows the model to
automatically discover which forecast horizons need more emphasis during
the cascade residual aggregation."""
import torch
import torch.nn as nn
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args     = model_args
        self.forecast_fc    = nn.Linear(hidden_dim, fk_dim)
        # Cascade特有: learnable per-step horizon weights
        total_steps = max(int(self.output_seq_len / model_args['gap']), 1)
        self.step_weights = nn.Parameter(torch.ones(total_steps))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes, num_feat]    = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            # RNN
            _gru    = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            # Cascade: apply LayerNorm from cascade's RNNLayer
            _gru_normed = rnn_layer.ln(_gru.squeeze(0)).unsqueeze(0)
            RNN_H   = torch.cat([RNN_H, _gru_normed], dim=0)
            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Transformer
            _Z  = transformer_layer(_gru_normed, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        # Cascade特有: apply learnable step weights (softmax-normalized)
        total_steps = predict.shape[1]
        w = torch.softmax(self.step_weights[:total_steps], dim=0)
        # Reshape for broadcasting: [1, steps, 1, 1]
        w = w.view(1, -1, 1, 1)
        predict = predict * w * total_steps  # scale to preserve magnitude

        if _CAS_DBG:
            print(f"[CAS:forecast@inh_forecast] steps={predict.shape[1]} "
                  f"weights={w.squeeze().detach().tolist()} "
                  f"range=[{predict.min().item():.4f},{predict.max().item():.4f}]",
                  file=sys.stderr)
        return predict
