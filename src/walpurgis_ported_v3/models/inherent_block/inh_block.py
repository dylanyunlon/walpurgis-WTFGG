"""
Inherent Block: positional encoding + GRU + Transformer + forecast/backcast.
Captures the inherent (non-diffusion) temporal patterns.
"""
import math
import sys

import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_DBG = ("--debug-inhblk" in sys.argv)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE identical to 'Attention Is All You Need'."""

    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.drop(x + self.pe[:x.size(0)])


class InhBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **kw):
        super().__init__()
        self.hidden_dim = hidden_dim

        # temporal encoder stack
        self.pos_enc = PositionalEncoding(hidden_dim, kw['dropout'])
        self.rnn     = RNNLayer(hidden_dim, kw['dropout'])
        self.msa     = TransformerLayer(hidden_dim, num_heads, kw['dropout'], bias)

        # branches
        self.fcast_head = Forecast(hidden_dim, forecast_hidden_dim, **kw)
        self.bcast_fc   = nn.Linear(hidden_dim, hidden_dim)
        self.res_decomp = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, signal):
        """
        signal: (B, L, N, D) — the inherent component from the previous layer.
        Returns (backcast_residual, forecast_hidden).
        """
        B, L, N, D = signal.shape

        # ── temporal encoding ──
        rnn_out = self.rnn(signal)            # (L, B*N, D)
        rnn_pe  = self.pos_enc(rnn_out)
        msa_out = self.msa(rnn_pe, rnn_pe, rnn_pe)

        if _DBG:
            print(f"[DBG:inhblk] signal=({B},{L},{N},{D})  "
                  f"rnn_out={tuple(rnn_out.shape)}  "
                  f"msa_mean={msa_out.mean().item():.4f}")

        # ── forecast branch ──
        fk = self.fcast_head(signal, rnn_out, msa_out,
                             self.msa, self.rnn, self.pos_enc)

        # ── backcast branch ──
        # reshape msa_out back to (B, L, N, D)
        msa_4d = msa_out.reshape(L, B, N, D).transpose(0, 1)
        bk = self.bcast_fc(msa_4d)
        residual = self.res_decomp(signal, bk)

        return residual, fk
