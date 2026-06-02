"""
Inherent Block: captures the residual (inherent) temporal patterns
via GRU → Positional Encoding → Multi-head Self-Attention,
plus forecast / backcast / residual decomposition.
"""

import math
import torch
import torch.nn as nn
import sys

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_DBG_INHBLK = ("--debug-inhblk" in sys.argv) or False


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (seq-first layout)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, X):
        """X : [L, *, D]"""
        X = X + self.pe[:X.size(0)]
        return self.drop(X)


class InhBlock(nn.Module):
    """
    Inherent block with GRU → PE → MSA temporal encoder,
    plus forecast / backcast / residual branches.
    """

    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.d_feat = hidden_dim
        self.d_hidden = hidden_dim

        # temporal encoder stack
        self.pos_enc = SinusoidalPE(hidden_dim, model_args['dropout'])
        self.rnn     = RNNLayer(hidden_dim, model_args['dropout'])
        self.attn    = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)

        # branches
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc     = nn.Linear(hidden_dim, hidden_dim)
        self.residual_link   = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, inherent_signal):
        """
        Parameters
        ----------
        inherent_signal : [B, L, N, D]

        Returns
        -------
        backcast_residual : [B, L, N, D]  — feed to next decouple layer
        forecast_hidden   : [B, H, N, fk_dim]
        """
        B, L, N, D = inherent_signal.shape

        # ── temporal encoding ──
        rnn_out = self.rnn(inherent_signal)          # [L, B*N, D]
        rnn_pe  = self.pos_enc(rnn_out)              # [L, B*N, D]
        attn_out = self.attn(rnn_pe, rnn_pe, rnn_pe) # [L, B*N, D]

        # ── forecast branch (AR roll-out) ──
        fk_hidden = self.forecast_branch(
            inherent_signal, rnn_out, attn_out,
            self.attn, self.rnn, self.pos_enc
        )

        # ── backcast branch ──
        # reshape attention output back: [L, B*N, D] → [B, L, N, D]
        attn_reshaped = attn_out.reshape(L, B, N, D).transpose(0, 1)
        bc_seq = self.backcast_fc(attn_reshaped)

        # ── residual decomposition ──
        residual_out = self.residual_link(inherent_signal, bc_seq)

        if _DBG_INHBLK:
            print(f"[DBG:inhblk] rnn_out_norm={rnn_out.norm().item():.4f}  "
                  f"attn_norm={attn_out.norm().item():.4f}  "
                  f"fk={tuple(fk_hidden.shape)}  "
                  f"residual={tuple(residual_out.shape)}")
        return residual_out, fk_hidden
