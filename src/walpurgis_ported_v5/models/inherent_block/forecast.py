import torch
import torch.nn as nn

# Delta vs upstream:
#   1. AR rollout uses LSTM state (hx, cx) instead of GRU hx only
#   2. forecast_fc → Linear + GELU + Linear (2-layer projection)

class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args

        # ── delta 2: 2-layer forecast head ──
        self.forecast_fc = nn.Sequential(
            nn.Linear(hidden_dim, fk_dim),
            nn.GELU(),
            nn.Linear(fk_dim, fk_dim),
        )

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]

        # ── delta 1: initialise LSTM cell state ──
        last_hx = RNN_H[-1]
        last_cx = torch.zeros_like(last_hx)

        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            # LSTM step
            last_hx, last_cx = rnn_layer.rnn_cell(
                predict[-1][0], (last_hx, last_cx))
            _rnn = last_hx.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _rnn], dim=0)

            if pe is not None:
                RNN_H = pe(RNN_H)

            _Z = transformer_layer(_rnn, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)                 # [T', B*N, D]
        predict = predict.reshape(-1, B, N, D)
        predict = predict.transpose(0, 1)                   # [B, T', N, D]
        predict = self.forecast_fc(predict)                  # delta 2
        return predict
