"""
InhBlock — Nightfall变体
算法改写:
  1. PositionalEncoding: 可学习phase offset (每个维度独立偏移)
  2. backcast分支: gated residual (sigmoid门控而非直接减法)
  3. PE中dropout后加LayerNorm
"""
import math
import torch
import torch.nn as nn
from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast
from walpurgis_nightfall import _dbg


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.post_ln = nn.LayerNorm(d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # 可学习phase offset: 每个维度独立
        self.phase_offset = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        pe_shifted = self.pe[:X.size(0)] + self.phase_offset
        X = X + pe_shifted
        X = self.dropout(X)
        X = self.post_ln(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        # gated residual: sigmoid门控
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = hidden_inherent_signal.shape
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)
        _dbg("inh_block.transformer_out", hidden_states_inh, "model")
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        # gated residual backcast
        gate = torch.sigmoid(self.gate_fc(hidden_states_inh))
        backcast_seq = gate * backcast_seq
        _dbg("inh_block.gate_mean", gate.mean(), "model")
        backcast_seq_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)
        return backcast_seq_res, forecast_hidden
