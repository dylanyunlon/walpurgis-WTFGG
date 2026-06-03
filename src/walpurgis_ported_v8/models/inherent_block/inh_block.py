import math
import torch
import torch.nn as nn
import sys

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_DBG = ("--dbg" in sys.argv)


class RotaryLikeEncoding(nn.Module):
    """算法改动: Rotary-like positional encoding
    原版: 标准 sinusoidal PE (加法式)
    改为: 类似 RoPE 的旋转编码
      - 仍然用 sin/cos 但以乘法方式注入:
        X_even = X_even * cos(theta) - X_odd * sin(theta)
        X_odd  = X_even * sin(theta) + X_odd * cos(theta)
      - 相对位置信息通过内积自然保持
      - 不需要额外参数
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        # [max_len, d_model/2]
        self.register_buffer('sin_table',
                             torch.sin(position * div_term))
        self.register_buffer('cos_table',
                             torch.cos(position * div_term))

    def forward(self, X):
        # X shape: [seq_len, batch*nodes, feat]
        seq_len = X.size(0)
        d = X.size(-1)
        half = d // 2
        cos_pe = self.cos_table[:seq_len, :half]  # [seq, half]
        sin_pe = self.sin_table[:seq_len, :half]
        # broadcast: [seq, 1, half]
        cos_pe = cos_pe.unsqueeze(1)
        sin_pe = sin_pe.unsqueeze(1)
        x_even = X[..., :half]
        x_odd = X[..., half:2*half]
        X_new = X.clone()
        X_new[..., :half] = x_even * cos_pe - x_odd * sin_pe
        X_new[..., half:2*half] = x_even * sin_pe + x_odd * cos_pe
        return X_new


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = RotaryLikeEncoding(hidden_dim)
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)

        # 算法改动: layer_scale — 可学习的缩放因子, 初始化为小值 (1e-4)
        # 让 backcast 分支初始贡献极小, 训练早期不干扰主路径
        self.layer_scale = nn.Parameter(
            torch.ones(hidden_dim) * 1e-4)

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = \
            hidden_inherent_signal.shape

        # RNN
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        # Rotary PE
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        # MSA with pre-norm + residual (inside TransformerLayer)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # forecast
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, self.pos_encoder)

        # backcast with layer_scale
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq = backcast_seq * self.layer_scale  # 算法改动
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        if _DBG:
            with torch.no_grad():
                ls_mean = self.layer_scale.mean().item()
                print(f"[DBG][InhBlock] layer_scale_mean={ls_mean:.6f}  "
                      f"forecast shape={list(forecast_hidden.shape)}  "
                      f"backcast_res std={backcast_seq_res.std().item():.5f}",
                      flush=True)
        return backcast_seq_res, forecast_hidden
