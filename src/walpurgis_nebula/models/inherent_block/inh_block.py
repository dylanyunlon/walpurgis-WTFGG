"""Nebula InhBlock: Fourier feature positional encoding + IndRNN + flash attention."""
import math, torch, torch.nn as nn, sys, os
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class FourierPositionalEncoding(nn.Module):
    """Random Fourier features for positional encoding.
    PE(t) = [sin(w_1*t), cos(w_1*t), ..., sin(w_d/2*t), cos(w_d/2*t)]
    where w_i are drawn from N(0, sigma^2) and frozen after init.
    This provides a richer, non-deterministic frequency basis vs fixed sinusoidal PE."""
    def __init__(self, d_model, dropout=0.1, sigma=10.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Random frequencies: frozen after init
        half_d = d_model // 2
        freqs = torch.randn(half_d) * sigma
        self.register_buffer('freqs', freqs)
        # Learnable amplitude scaling per frequency
        self.amplitude = nn.Parameter(torch.ones(d_model))

    def forward(self, X):
        """X: [seq_len, batch*nodes, d_model]"""
        seq_len = X.size(0)
        device = X.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # [S, 1]
        # Fourier features
        angles = t * self.freqs.unsqueeze(0)  # [S, d/2]
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [S, d]
        # Handle odd d_model
        if pe.shape[-1] < X.shape[-1]:
            pe = torch.cat([pe, torch.zeros(seq_len, X.shape[-1] - pe.shape[-1], device=device)], dim=-1)
        elif pe.shape[-1] > X.shape[-1]:
            pe = pe[:, :X.shape[-1]]
        pe = pe.unsqueeze(1) * self.amplitude  # [S, 1, d]
        X = X + pe
        return self.dropout(X)


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # Nebula: Fourier random feature PE
        self.pos_encoder = FourierPositionalEncoding(hidden_dim, model_args['dropout'], sigma=10.0)
        # Nebula: IndRNN replaces GRU
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        # Nebula: flash attention pattern replaces standard MSA
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # Highway-style backcast (reuse from dif_block concept)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_gate = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, num_nodes, num_feat] = hidden_inherent_signal.shape
        # IndRNN
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        # Fourier PE
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        # Flash attention
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)
        # Forecast branch
        forecast_hidden = self.forecast_block(hidden_inherent_signal, hidden_states_rnn, hidden_states_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)
        # Nebula: gated backcast (highway-like)
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        bc_h = torch.tanh(self.backcast_fc(hidden_states_inh))
        bc_g = torch.sigmoid(self.backcast_gate(hidden_states_inh))
        backcast_seq = bc_g * bc_h + (1.0 - bc_g) * hidden_states_inh
        backcast_seq_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)
        if _NEB_DBG:
            print(f"[NEB:block@inh_block] forecast={list(forecast_hidden.shape)} gate_mean={bc_g.mean().item():.4f}", file=sys.stderr)
        return backcast_seq_res, forecast_hidden
