import torch
import torch.nn as nn
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:inhfc:{tag}] shape={list(val.shape)} range=[{val.min().item():.4f},{val.max().item():.4f}]", file=sys.stderr)

class Forecast(nn.Module):
    """upstream: 直接cat predict
    solstice: 指数移动平均平滑多步预测, 可学习EMA衰减"""
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # solstice: 可学习EMA衰减因子
        self.ema_decay = nn.Parameter(torch.tensor(0.9))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step in range(n_steps):
            # solstice: LSTM cell — 需要传h和c
            # rnn_layer.lstm_cell expects (input, (hx, cx))
            # 我们用隐状态RNN_H[-1]作为hx, 初始化cx为0
            _h = rnn_layer.lstm_cell(
                predict[-1][0],
                (RNN_H[-1], torch.zeros_like(RNN_H[-1]))
            )[0]  # 取h, 忽略c
            _h = _h.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _h], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_h, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)

        # solstice: EMA平滑预测序列
        decay = torch.clamp(self.ema_decay, 0.5, 0.99)
        n = predict.shape[1]
        if n > 1:
            smoothed = [predict[:, 0:1, :, :]]
            for t in range(1, n):
                s = decay * smoothed[-1] + (1 - decay) * predict[:, t:t+1, :, :]
                smoothed.append(s)
            predict = torch.cat(smoothed, dim=1)
        _sdbg("ema_decay", decay)

        predict = self.forecast_fc(predict)
        _sdbg("inh_forecast", predict)
        return predict
