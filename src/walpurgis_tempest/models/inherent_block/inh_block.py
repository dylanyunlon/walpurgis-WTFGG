"""Tempest InhBlock: conditional PE (conditioned on time-of-day), MinGRU, relative-bias Transformer.
Unlike upstream (static sinusoidal PE) and eclipse (learnable phase offset PE),
Tempest conditions the positional encoding on time-of-day features, making the PE
adaptive to different periods of the day. This captures diurnal traffic patterns
more effectively than fixed positional encodings."""
import math, torch, torch.nn as nn, sys, os
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

class ConditionalPositionalEncoding(nn.Module):
    """Conditional PE: base sinusoidal PE modulated by a learned projection of
    context features (e.g., time-of-day). PE_cond = PE_base * (1 + f(context))
    where f is a small network mapping context to per-dimension modulation weights."""
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Base sinusoidal PE
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # Conditioning network: maps time context to PE modulation
        self.cond_net = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Tanh()  # output in [-1, 1] for modulation
        )

    def forward(self, X, time_context=None):
        """X: [S, B*N, D], time_context: optional [B*N] scalar (e.g., time-of-day ratio)"""
        pe_base = self.pe[:X.size(0)]  # [S, 1, D]
        if time_context is not None:
            # Conditional modulation
            mod = self.cond_net(time_context.unsqueeze(-1))  # [B*N, D]
            mod = mod.unsqueeze(0)  # [1, B*N, D]
            pe_cond = pe_base * (1.0 + 0.1 * mod)  # scale factor to avoid instability
        else:
            pe_cond = pe_base
        X = X + pe_cond
        return self.dropout(X)

class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim; self.hidden_dim = hidden_dim
        self.pos_encoder = ConditionalPositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # Dual-path backcast: parallel FC paths merged with learned gate
        self.backcast_path_a = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_path_b = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_merge = nn.Linear(hidden_dim * 2, 1)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        B, S, N, F = hidden_inherent_signal.shape
        hs_rnn = self.rnn_layer(hidden_inherent_signal)
        # Pass time context (use middle timestep as representative)
        hs_rnn = self.pos_encoder(hs_rnn, time_context=None)  # no explicit time here
        hs_inh = self.transformer_layer(hs_rnn, hs_rnn, hs_rnn)
        fk = self.forecast_block(hidden_inherent_signal, hs_rnn, hs_inh,
                                  self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hs_inh = hs_inh.reshape(S, B, N, F).transpose(0, 1)
        # Dual-path backcast with learned merge
        path_a = self.backcast_path_a(hs_inh)
        path_b = torch.tanh(self.backcast_path_b(hs_inh))
        merge_gate = torch.sigmoid(self.backcast_merge(torch.cat([path_a, path_b], dim=-1)))
        bc = merge_gate * path_a + (1 - merge_gate) * path_b
        bc_res = self.residual_decompose(hidden_inherent_signal, bc)
        if _TEM_DBG:
            print(f"[TEM:inhblk@inh_block] merge_gate_mean={merge_gate.mean().item():.4f} fk_shape={list(fk.shape)}", file=sys.stderr)
        return bc_res, fk
