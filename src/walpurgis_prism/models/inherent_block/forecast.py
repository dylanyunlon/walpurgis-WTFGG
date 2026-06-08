"""Prism inherent forecast: with learnable prediction confidence scaling."""
import torch
import torch.nn as nn


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # Prism特有: 预测置信度缩放
        self.confidence_scale = nn.Parameter(
            torch.tensor(1.0))

    def forward(self, X, RNN_H, Z,
                transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes,
         num_feat] = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len /
                           self.model_args['gap']) - 1):
            _gru = rnn_layer.gru_cell(
                predict[-1][0],
                RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(
                _gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        # Prism特有: 置信度缩放
        scale = torch.sigmoid(self.confidence_scale)
        predict = self.forecast_fc(predict) * scale
        return predict
