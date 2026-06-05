"""
forecast.py (inherent) — CardGame Inherent Forecast
算法改写 (vs upstream):
  - 新增可学习步长衰减: 越远的预测步权重越小
  - 新增ALiBi位置偏移: 替代sinusoidal PE, 直接偏移attention score
"""
import os
import sys
import math
import torch
import torch.nn as nn

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="InhForecast"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class Forecast(nn.Module):
    """CardGame Inherent Forecast: learnable step decay + ALiBi"""

    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

        # CardGame: 可学习步长衰减 (每个预测步的衰减系数)
        num_steps = max(int(self.output_seq_len / model_args['gap']), 1)
        self.step_decay = nn.Parameter(torch.ones(num_steps))

        # CardGame: ALiBi slope参数
        self.alibi_slope = nn.Parameter(
            torch.tensor([1.0 / (2 ** (i + 1))
                          for i in range(num_steps)]))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        _dbg("input.X", X)
        _dbg("input.Z", Z)

        [batch_size, _, num_nodes, num_feat] = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for step in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            # RNN
            _gru = rnn_layer.gru_cell(
                predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            # Positional Encoding (ALiBi bias applied here)
            if pe is not None:
                RNN_H_pe = pe(RNN_H)
            else:
                RNN_H_pe = RNN_H

            # CardGame: ALiBi位置偏移
            # 在transformer attention之前对K加上距离偏移
            alibi_bias = self.alibi_slope[step] * torch.arange(
                RNN_H_pe.shape[0], device=RNN_H_pe.device).float()
            alibi_bias = alibi_bias.unsqueeze(-1).unsqueeze(-1)
            RNN_H_alibi = RNN_H_pe - alibi_bias * 0.01  # small bias

            _Z = transformer_layer(_gru, K=RNN_H_alibi, V=RNN_H_alibi)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)

        # CardGame: 可学习步长衰减
        decay = torch.sigmoid(self.step_decay)
        num_pred_steps = predict.shape[1]
        decay_weights = decay[:num_pred_steps].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        predict = predict * decay_weights
        _dbg("step_decay_weights", decay[:num_pred_steps])

        predict = self.forecast_fc(predict)
        _dbg("forecast_output", predict)
        return predict
