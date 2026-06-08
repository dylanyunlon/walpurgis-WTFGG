"""
Forecast (inherent branch) — Perihelion变体
适配Flash-Chunk Transformer + SwiGLU接口
  原版(penumbra): MinGRU手动步进 + CrossAttention逐步预测
  Perihelion: GRU步进保持兼容, 用ChunkedAttention+SwiGLU做单步推进
             每个预测步先GRU更新hidden, 再用TransformerLayer做上下文融合
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

        _dbg("inh_forecast.init_token",
             predict[-1], "inherent")

        for step in range(
                int(self.output_seq_len
                    / self.model_args['gap']) - 1):
            # GRU一步: 手动update (保持与penumbra兼容的接口)
            prev = predict[-1][0]  # [B*N, D]
            combined = torch.cat(
                [RNN_H[-1], prev], dim=-1)
            z = torch.sigmoid(rnn_layer.W_z(combined))
            r = torch.sigmoid(rnn_layer.W_r(combined))
            h_tilde = torch.tanh(
                rnn_layer.W_h(
                    torch.cat([r * RNN_H[-1], prev],
                              dim=-1)))
            _gru = ((1 - z) * RNN_H[-1]
                    + z * h_tilde).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Flash-Chunk Transformer一步:
            # 用完整RNN历史做context
            _Z = transformer_layer(_gru)
            predict.append(_Z)

            _dbg(f"inh_forecast.step_{step}",
                 _Z, "inherent")

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("inh_forecast.output",
             predict, "inherent")
        return predict
