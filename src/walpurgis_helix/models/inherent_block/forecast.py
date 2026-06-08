import torch
import torch.nn as nn

class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args     = model_args

        self.forecast_fc    = nn.Linear(hidden_dim, fk_dim)
        # Helix特有: 螺旋相位注入forecast — 不同预测步加不同相位
        self.step_phase = nn.Parameter(
            torch.randn(int(self.output_seq_len / model_args['gap'])) * 0.01)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes, num_feat]    = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        step_idx = 0
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            # RNN
            _gru    = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H   = torch.cat([RNN_H, _gru], dim=0)
            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Transformer
            _Z  = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
            step_idx += 1

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        # Helix特有: 对每个预测步加螺旋相位偏置
        n_steps = predict.shape[1]
        phase = torch.sigmoid(self.step_phase[:n_steps])
        phase = phase.view(1, n_steps, 1, 1)
        predict = predict * (0.9 + 0.2 * phase)
        return predict
