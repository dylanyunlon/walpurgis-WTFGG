"""Flux Forecast (diffusion branch): 渐进式AR预测.
与upstream(直接AR展开)和vortex(同upstream)不同,
Flux在AR展开时加入渐进式衰减: 每一步预测的置信度随步数衰减,
并对历史片段施加因果衰减权重, 使远期预测更保守."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class Forecast(nn.Module):
    def __init__(self, hidden_dim,
                 forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(
            hidden_dim, forecast_hidden_dim)
        self.model_args = model_args
        # Flux: 渐进式步进衰减 — 可学习
        n_steps = max(
            int(self.output_seq_len /
                self.model_args['gap']) - 1, 1)
        self.step_confidence = nn.Parameter(
            torch.linspace(1.0, 0.7, n_steps))

    def forward(self, gated_history_data,
                hidden_states_dif, localized_st_conv,
                dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(
            hidden_states_dif[:, -1, :, :].unsqueeze(1))
        n_steps = int(
            self.output_seq_len /
            self.model_args['gap']) - 1
        for step_idx in range(n_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            new_pred = localized_st_conv(
                _1, dynamic_graph, static_graph)
            # Flux: 渐进式衰减 — 远期步骤置信度更低
            confidence = torch.sigmoid(
                self.step_confidence[
                    min(step_idx,
                        len(self.step_confidence) - 1)])
            new_pred = new_pred * confidence
            predict.append(new_pred)
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        if _FX_DBG:
            print(f"[FX:dif_forecast] n_steps={n_steps} "
                  f"conf_range=["
                  f"{torch.sigmoid(self.step_confidence[0]).item():.3f},"
                  f"{torch.sigmoid(self.step_confidence[-1]).item():.3f}]",
                  file=sys.stderr)
        return predict
