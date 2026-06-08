"""
Forecast (inherent branch) — Transit变体
适配S4 + GatedAttention接口
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

    def forward(self, X, S4_H, Z, transformer_layer,
                s4_layer, pe):
        batch_size, _, num_nodes, num_feat = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(
                int(self.output_seq_len
                    / self.model_args['gap']) - 1):
            prev = predict[-1]  # [1, B*N, D]
            # S4一步: 用状态空间模型推进
            s4_out = s4_layer(prev)  # [1, B*N, D]
            S4_H = torch.cat([S4_H, s4_out], dim=0)
            if pe is not None:
                S4_H = pe(S4_H)
            # 门控注意力一步
            _Z = transformer_layer(s4_out, S4_H, S4_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("inh_forecast.output",
             predict, "inherent")
        return predict
