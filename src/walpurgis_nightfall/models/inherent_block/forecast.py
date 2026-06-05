"""
Forecast (inherent) — Nightfall变体
算法改写:
  1. 自回归预测中加scheduled sampling概率 (训练时随机用真值替代预测)
  2. AR步骤间加dropout (防止error accumulation)
"""
import torch
import torch.nn as nn
from walpurgis_nightfall import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # AR dropout
        self.ar_dropout = nn.Dropout(model_args.get('dropout', 0.3) * 0.5)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        batch_size, _, num_nodes, num_feat = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        num_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step_i in range(num_steps):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            # AR dropout: 防止错误积累
            if self.training:
                _Z = self.ar_dropout(_Z)
            predict.append(_Z)
            if step_i == 0:
                _dbg("inh_forecast.ar_step0", _Z, "model")
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict
