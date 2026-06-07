import torch
import torch.nn as nn
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:diffc:{tag}] shape={list(val.shape)} range=[{val.min().item():.4f},{val.max().item():.4f}]", file=sys.stderr)


class ScaleNorm(nn.Module):
    """solstice: ScaleNorm"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return self.g * x / norm


class Forecast(nn.Module):
    """upstream: 直接history padding
    solstice: 反射padding + ScaleNorm后置"""
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        # solstice: ScaleNorm后置
        self.post_sn = ScaleNorm(forecast_hidden_dim)
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
                avail = history[:, -deficit:, :, :]
                pad = torch.flip(avail, dims=[1])
                recent = [pad] + recent
            recent = torch.cat(recent, dim=1)
            predict.append(localized_st_conv(recent, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        # solstice: ScaleNorm
        predict = self.post_sn(predict)
        _sdbg("dif_forecast", predict)
        return predict
