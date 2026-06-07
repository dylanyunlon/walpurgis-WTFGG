import math
import torch
import torch.nn as nn
import sys, os

from walpurgis_equinox.models.decouple.residual_decomp import ResidualDecomp
from walpurgis_equinox.models.inherent_block.inh_model import RNNLayer, TransformerLayer
from walpurgis_equinox.models.inherent_block.forecast import Forecast

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:inhblk:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class PositionalEncoding(nn.Module):
    """upstream: 固定正弦PE
    equinox: 可学习phase offset + 频率缩放"""
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # equinox: 可学习phase offset + 频率缩放
        self.phase_offset = nn.Parameter(torch.zeros(1, 1, d_model))
        self.freq_scale = nn.Parameter(torch.ones(1, 1, d_model))

    def forward(self, X):
        X = X + self.pe[:X.size(0)] * self.freq_scale + self.phase_offset
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    """upstream: 直接FC backcast
    equinox: DenseNet式门控残差 — 聚合所有中间特征"""
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)

        # equinox: DenseNet式dense connection
        # 聚合 input + rnn_out + transformer_out 三路特征
        self.dense_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dense_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.bc_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        B, L, N, D = hidden_inherent_signal.shape

        hidden_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_rnn_pe = self.pos_encoder(hidden_rnn)
        hidden_inh = self.transformer_layer(hidden_rnn_pe, hidden_rnn_pe, hidden_rnn_pe)

        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_rnn, hidden_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)

        hidden_inh = hidden_inh.reshape(L, B, N, D).transpose(0, 1)
        hidden_rnn_reshaped = hidden_rnn.reshape(L, B, N, D).transpose(0, 1)

        # equinox: DenseNet式dense connection — 聚合input + rnn + transformer
        dense_cat = torch.cat([
            hidden_inherent_signal,
            hidden_rnn_reshaped,
            hidden_inh
        ], dim=-1)  # [B, L, N, 3D]

        gate = torch.sigmoid(self.dense_gate(dense_cat))
        dense_fused = gate * self.dense_proj(dense_cat)
        backcast_seq = self.bc_fc(dense_fused)

        _edbg("dense_gate", gate)
        _edbg("dense_fused", dense_fused)
        _edbg("inh_backcast", backcast_seq)

        backcast_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)
        return backcast_res, forecast_hidden
