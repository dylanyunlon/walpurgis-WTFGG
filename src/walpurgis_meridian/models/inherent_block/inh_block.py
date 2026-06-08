"""Meridian InhBlock — learnable frequency positional encoding.
Changes vs upstream:
  - FrequencyPE: learnable frequency bands (upstream: fixed sinusoidal)
  - Uses HighwayGRU and RelPosTransformer
"""
import math
import torch
import torch.nn as nn
import sys, os

from walpurgis_meridian.models.decouple.residual_decomp import ResidualDecomp
from walpurgis_meridian.models.inherent_block.inh_model import HighwayGRULayer, RelPosTransformerLayer
from walpurgis_meridian.models.inherent_block.forecast import Forecast

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class FrequencyPositionalEncoding(nn.Module):
    """Learnable frequency bands for positional encoding.
    Unlike fixed sinusoidal PE, the frequency components are trainable."""
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # learnable frequency parameters
        self.freq = nn.Parameter(torch.randn(d_model // 2) * 0.01)
        self.phase = nn.Parameter(torch.zeros(d_model // 2))
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, X):
        seq_len = X.size(0)
        positions = torch.arange(seq_len, dtype=torch.float32, device=X.device).unsqueeze(1)
        freq = torch.exp(self.freq)
        angles = positions * freq.unsqueeze(0) + self.phase.unsqueeze(0)
        pe = torch.zeros(seq_len, 1, self.d_model, device=X.device)
        pe[:, 0, 0::2] = torch.sin(angles)
        pe[:, 0, 1::2] = torch.cos(angles[:, :self.d_model // 2]) if self.d_model % 2 == 0 \
            else torch.cos(angles[:, :self.d_model // 2])
        X = X + pe[:X.size(0)]
        X = self.dropout(X)
        if _DBG:
            print(f"[MER:freq_pe] freq_range=[{freq.min().item():.4f},{freq.max().item():.4f}] "
                  f"pe_norm={pe.norm().item():.4f}", file=sys.stderr)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = FrequencyPositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = HighwayGRULayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = RelPosTransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)

        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)

        if _DBG:
            print(f"[MER:inh_block] fk_norm={forecast_hidden.detach().norm().item():.4f} "
                  f"bc_norm={backcast_seq_res.detach().norm().item():.4f}",
                  file=sys.stderr)

        return backcast_seq_res, forecast_hidden
