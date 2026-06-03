"""Inherent block — RoPE-style positional encoding + layer diagnostics.

Changes
-------
1. ``PositionalEncoding`` replaced with ``RotaryPositionalEncoding``:
   instead of additive sinusoidal PE, applies rotary multiplication to
   even/odd feature pairs.  RoPE naturally decays with relative distance,
   which suits auto-regressive forecast unrolling better than absolute PE.
2. ``forward`` prints per-layer hidden state norms when debug is on,
   so you can track signal magnitude through the inherent stack.
"""

import math
import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast
from walpurgis_ported_v6 import _dbg


class RotaryPositionalEncoding(nn.Module):
    """Simplified RoPE: rotate even/odd feature pairs by position-dependent angles."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pre-compute angle frequencies
        half_d = d_model // 2
        freqs = torch.exp(
            torch.arange(0, half_d, dtype=torch.float32)
            * (-math.log(10000.0) / half_d)
        )
        positions = torch.arange(max_len, dtype=torch.float32)
        # (max_len, half_d)
        angles = torch.outer(positions, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, X):
        # X: (seq_len, batch_stuff, d_model)
        T = X.size(0)
        d = X.size(-1)
        half = d // 2
        x1, x2 = X[..., :half], X[..., half:2*half]
        cos = self.cos_cached[:T].unsqueeze(1)   # (T, 1, half)
        sin = self.sin_cached[:T].unsqueeze(1)
        # rotate
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        out = torch.cat([r1, r2], dim=-1)
        # if d is odd, append the last feature unchanged
        if d % 2 == 1:
            out = torch.cat([out, X[..., -1:]], dim=-1)
        return self.dropout(out)


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = RotaryPositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        B, T, N, D = hidden_inherent_signal.shape

        _dbg("InhBlock.input", hidden_inherent_signal)

        # RNN
        h_rnn = self.rnn_layer(hidden_inherent_signal)
        _dbg("InhBlock.post_rnn", h_rnn)

        # Rotary PE
        h_rnn = self.pos_encoder(h_rnn)

        # Transformer (pre-norm is inside TransformerLayer)
        h_inh = self.transformer_layer(h_rnn, h_rnn, h_rnn)
        _dbg("InhBlock.post_transformer", h_inh)

        # forecast
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, h_rnn, h_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)

        # backcast + residual
        h_inh = h_inh.reshape(T, B, N, D).transpose(0, 1)
        backcast = self.backcast_fc(h_inh)
        backcast_res = self.residual_decompose(
            hidden_inherent_signal, backcast)

        return backcast_res, forecast_hidden
