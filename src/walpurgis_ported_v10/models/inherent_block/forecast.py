import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

_TAG = "inhfc"


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args

        # 改动3: FC前加GELU + Dropout
        self.pre_act = nn.GELU()
        self.pre_drop = nn.Dropout(0.1)
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

        # 改动1: 可学习步长衰减 γ — upstream 所有 AR 步等权
        # exp(-γ * step), γ 可学习, init γ=0.05
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))  # exp(-3) ≈ 0.05

    def _apply_rope(self, x, seq_dim=0):
        """改动2: 简化版 RoPE — upstream 用标准 sincos PE.
        RoPE 对 Q/K 的偶数/奇数维分别做旋转, 这里近似:
        对 hidden 前半 cos 旋转, 后半 sin 旋转."""
        seq_len = x.shape[seq_dim]
        device = x.device
        D = x.shape[-1]
        half_d = D // 2

        pos = torch.arange(seq_len, device=device).float().unsqueeze(-1)
        freq = torch.exp(-torch.arange(half_d, device=device).float()
                         * (4.0 / half_d))
        angles = pos * freq  # (seq_len, half_d)

        # expand to match x shape
        while angles.dim() < x.dim():
            angles = angles.unsqueeze(1)

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        x1, x2 = x[..., :half_d], x[..., half_d:2*half_d]
        rotated = torch.cat([x1 * cos_a - x2 * sin_a,
                              x1 * sin_a + x2 * cos_a], dim=-1)
        if D % 2 == 1:
            rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
        return rotated

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, _, N, D = X.shape
        gamma = torch.exp(self.log_gamma).clamp(min=0.01, max=0.5)

        predict = [Z[-1, :, :].unsqueeze(0)]
        n_steps = int(self.output_seq_len / self.model_args['gap']) - 1

        _dbg(_TAG, "ar_start", gamma=gamma, n_steps=n_steps)

        for step_i in range(n_steps):
            _gru = rnn_layer.gru_cell(
                predict[-1][0], RNN_H[-1]).unsqueeze(0)
            # RMSNorm from rnn_layer
            _gru_normed = rnn_layer.step_norm(_gru.squeeze(0)).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru_normed], dim=0)

            # 改动2: RoPE 替代 sincos PE
            RNN_H_rope = self._apply_rope(RNN_H, seq_dim=0)

            _Z = transformer_layer(_gru_normed, K=RNN_H_rope, V=RNN_H_rope)

            # 改动1: 步长衰减
            decay = torch.exp(-gamma * (step_i + 1))
            _Z = _Z * decay

            predict.append(_Z)

            _dbg(_TAG, f"step_{step_i}", decay=decay.item(), _Z=_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)

        # 改动3: GELU + Dropout before FC
        predict = self.pre_act(predict)
        predict = self.pre_drop(predict)
        predict = self.forecast_fc(predict)

        _dbg(_TAG, "output", predict=predict)
        return predict
