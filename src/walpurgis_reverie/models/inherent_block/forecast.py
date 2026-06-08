import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

_TAG = "inh_forecast"


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
        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            x_t = predict[-1][0]
            # MinGRU step
            hx = RNN_H[-1]
            z_t = torch.sigmoid(
                rnn_layer.W_z(torch.cat([x_t, hx], dim=-1)))
            h_candidate = torch.tanh(rnn_layer.W_h(x_t))
            _gru = ((1 - z_t) * hx + z_t * h_candidate).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        _dbg(f"{_TAG}/forecast_out", predict, _TAG)
        return predict
