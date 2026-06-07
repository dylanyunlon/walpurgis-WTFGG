import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:diffc:{tag}] shape={list(val.shape)} range=[{val.min().item():.4f},{val.max().item():.4f}]", file=sys.stderr)

class Forecast(nn.Module):
    """upstream: 直接history padding
    solstice: 零填充padding + LayerNorm后置"""
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        # solstice: LayerNorm后置稳定forecast
        self.post_ln = nn.LayerNorm(forecast_hidden_dim)
        self.model_args = model_args

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                deficit = self.k_t - len(recent)
                # upstream: 用history尾部padding
                # solstice: 零填充 — 避免引入历史偏差
                B, _, N, D = predict[-1].shape
                pad = torch.zeros(B, deficit, N, D, device=predict[-1].device)
                recent = [pad] + recent
            recent = torch.cat(recent, dim=1)
            predict.append(localized_st_conv(recent, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        # solstice: LayerNorm
        predict = self.post_ln(predict)
        _adbg("dif_forecast", predict)
        return predict
