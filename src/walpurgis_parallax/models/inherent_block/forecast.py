"""
Forecast (inherent branch) — Parallax变体 (M054)
适配xLSTM和CrossAttention接口

与Penumbra差异:
  - 自回归步骤用xLSTM的显式门控更新(指数遗忘+归一化器)
  - 位置编码使用PositionalInterpolation
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
        self.hidden_dim = hidden_dim

    def forward(self, X, RNN_H, Z, transformer_layer,
                rnn_layer, pe):
        batch_size, _, num_nodes, num_feat = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]

        # 自回归xLSTM单步: 使用rnn_layer的权重做手动更新
        # xLSTM: f=exp(W_f), i=sigmoid(W_i), o=sigmoid(W_o)
        # 初始化normalizer为1 (首次无积累)
        h_prev = RNN_H[-1]  # [B*N, D]
        c_state = torch.zeros_like(h_prev)
        n_state = torch.ones_like(h_prev)

        for step_i in range(
                int(self.output_seq_len
                    / self.model_args['gap']) - 1):
            prev = predict[-1][0]  # [B*N, D]
            combined = torch.cat([h_prev, prev], dim=-1)

            # xLSTM gates
            i_t = torch.sigmoid(rnn_layer.W_i(combined))
            f_raw = rnn_layer.W_f(combined)
            f_t = torch.exp(torch.clamp(
                f_raw, max=rnn_layer._exp_clip))
            o_t = torch.sigmoid(rnn_layer.W_o(combined))
            c_tilde = torch.tanh(rnn_layer.W_c(prev))

            # 指数遗忘 + 归一化
            c_state = f_t * c_state + i_t * c_tilde
            n_state = f_t * n_state + i_t
            c_normed = c_state / (n_state + 1e-8)

            # 记忆混合
            mix = torch.sigmoid(rnn_layer.W_mix(combined))
            c_mixed = (mix * c_normed
                       + (1 - mix) * torch.tanh(c_state))
            h_new = o_t * torch.tanh(c_mixed)

            _gru = h_new.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            h_prev = h_new

            if pe is not None:
                RNN_H = pe(RNN_H)
            # Cross-attention一步
            _Z = transformer_layer(
                _gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("inh_forecast.output",
             predict, "inherent")
        _dbg("inh_forecast.steps",
             len(predict), "inherent")
        return predict
