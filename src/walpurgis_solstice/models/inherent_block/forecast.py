import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:inhfc:{tag}] shape={list(val.shape)} range=[{val.min().item():.4f},{val.max().item():.4f}]", file=sys.stderr)

class Forecast(nn.Module):
    """upstream: 直接cat predict
    solstice: 余弦相似度步长加权 — 每步预测与初始锚点的余弦相似度作权重"""
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        # solstice: 可学习锚向量, 用于余弦加权
        self.anchor = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        for step in range(n_steps):
            # solstice: LSTM cell expects (hx, cx); pass hx for hidden, zeros for cx
            _h = rnn_layer.lstm_cell(predict[-1][0], (RNN_H[-1], torch.zeros_like(RNN_H[-1])))
            _h_out = _h[0]  # hidden state from LSTM
            _h_out = rnn_layer.pn(_h_out)
            _h_out = _h_out.unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _h_out], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_h_out, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)

        # solstice: 余弦相似度步长加权
        anchor = self.anchor.expand(1, 1, D).to(predict.device)
        n = predict.shape[1]
        step_vecs = predict.mean(dim=2)  # [B, n, D]
        cos_sim = F.cosine_similarity(step_vecs, anchor.expand(step_vecs.shape[0], n, -1), dim=-1)
        weights = F.softmax(cos_sim, dim=-1)  # [B, n]
        weights = weights.unsqueeze(2).unsqueeze(3)  # [B, n, 1, 1]
        predict = predict * weights * n  # rescale保持期望
        _adbg("cosine_weights", weights.squeeze())

        predict = self.forecast_fc(predict)
        _adbg("inh_forecast", predict)
        return predict
