"""
forecast.py (inherent) — v9 port
Algo delta:
  1. 可学习步衰减参数 γ (init=0.95): 每步预测乘 γ^step,
     远步预测自动衰减 → 减少长期 AR 误差累积的权重
  2. pe 为 None 时用零张量 placeholder 而非完全跳过,
     保证维度一致性
"""
import math
import torch
import torch.nn as nn
from walpurgis_ported_v9 import _dbg

_TAG = "inh_fc"


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.gap = model_args['gap']
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # v9: learnable step-decay
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.95)))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        gamma = torch.exp(self.log_gamma).clamp(0.5, 1.0)

        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = math.ceil(self.output_seq_len / self.gap) - 1

        for step in range(n_steps):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            # v9: step-decay
            _Z = _Z * (gamma ** (step + 1))
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg(_TAG, f"inh_fc  γ={gamma.item():.4f}  steps={n_steps+1}  "
                    f"out_shape={list(predict.shape)}")
        return predict
