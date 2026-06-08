"""Flux inherent forecast: 渐进式AR预测.
与upstream(直接展开)和vortex(同upstream)不同,
Flux在inherent forecast的AR展开中也加入置信度衰减,
并在Transformer的K/V中只保留最近window步,
模拟流式推理中有限的历史窗口."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # Flux: 流式推理窗口大小
        self._stream_kv_window = 8

    def forward(self, X, RNN_H, Z, transformer_layer,
                rnn_layer, pe):
        [batch_size, _, num_nodes,
         num_feat] = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        n_forecast_steps = int(
            self.output_seq_len /
            self.model_args['gap']) - 1
        for step_idx in range(n_forecast_steps):
            # RNN
            _gru = rnn_layer.gru_cell(
                predict[-1][0],
                RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Flux: 流式窗口 — K/V只用最近window步
            kv_start = max(
                0, RNN_H.shape[0] -
                self._stream_kv_window)
            RNN_H_windowed = RNN_H[kv_start:]
            # Transformer with windowed K/V
            _Z = transformer_layer(
                _gru, K=RNN_H_windowed,
                V=RNN_H_windowed)
            predict.append(_Z)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        if _FX_DBG:
            print(f"[FX:inh_forecast] steps="
                  f"{n_forecast_steps} kv_window="
                  f"{self._stream_kv_window}",
                  file=sys.stderr)
        return predict
