import torch
import torch.nn as nn
import sys

_DBG = ("--dbg" in sys.argv)


class Forecast(nn.Module):
    """算法改动: AR loop dropout
    原版: 自回归循环中无 regularization, 误差自由积累
    改为: 每个 AR step 对 RNN hidden state 施加 dropout,
          且 dropout rate 随步数线性增大 (越远的预测加更强 regularization)
    """

    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.base_dropout = model_args.get('dropout', 0.1)
        self.total_steps = int(
            self.output_seq_len / self.model_args['gap']) - 1

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        batch_size, _, num_nodes, num_feat = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for step in range(self.total_steps):
            _gru = rnn_layer.gru_cell(
                predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            if pe is not None:
                if RNN_H.size(0) <= 5000:
                    RNN_H = pe(RNN_H)

            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)

            # 算法改动: step-progressive dropout
            step_drop_rate = min(
                self.base_dropout * (1.0 + step / max(self.total_steps, 1)),
                0.5)
            if self.training:
                _Z = torch.nn.functional.dropout(
                    _Z, p=step_drop_rate, training=True)

            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        if _DBG:
            with torch.no_grad():
                print(f"[DBG][InhForecast] predict shape={list(predict.shape)}  "
                      f"ar_steps={self.total_steps}  "
                      f"base_drop={self.base_dropout}", flush=True)
        return predict
