import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:inhfc:{tag}] shape={list(val.shape)} range=[{val.min().item():.4f},{val.max().item():.4f}]", file=sys.stderr)

class Forecast(nn.Module):
    """upstream: 直接cat predict
    aurora: 步长衰减权重 w=exp(-decay*step), 可学习decay"""
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # aurora: 可学习步长衰减率
        self.step_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step in range(n_steps):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1])
            _gru = rnn_layer.ln(_gru)
            _gru = _gru.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)

        # aurora: 步长衰减权重
        decay = torch.clamp(self.step_decay, min=0.01, max=1.0)
        n = predict.shape[1]
        weights = torch.exp(-decay * torch.arange(n, dtype=torch.float32, device=predict.device))
        weights = weights / weights.sum()
        weights = weights.view(1, n, 1, 1)
        predict = predict * weights * n  # rescale保持期望
        _adbg("step_weights", weights.squeeze())

        predict = self.forecast_fc(predict)
        _adbg("inh_forecast", predict)
        return predict
