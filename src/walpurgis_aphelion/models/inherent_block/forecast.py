import torch
import torch.nn as nn


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
            # Aphelion: 用retention的forecast_cell做递归预测
            prev = predict[-1][0]
            if rnn_layer._last_hidden is not None:
                _h = torch.tanh(rnn_layer._forecast_cell(prev) + rnn_layer._last_hidden)
            else:
                _h = torch.tanh(rnn_layer._forecast_cell(prev))
            rnn_layer._last_hidden = _h.detach()
            _out = _h.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _out], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_out, K=RNN_H, V=RNN_H)
            predict.append(_Z)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict
