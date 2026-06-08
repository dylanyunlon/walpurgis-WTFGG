"""Rift InhBlock: inherent block with Rift-specific modifications.
- PositionalEncoding with learned scaling (Rift特有)
- Backcast uses SiLU activation instead of identity (Rift特有)
"""
import math
import torch
import torch.nn as nn
import sys, os

from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast

_RF_DBG = os.environ.get('RIFT_DEBUG', '0') == '1'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # Rift特有: 可学习的PE缩放系数
        self.pe_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        X = X + self.pe_scale * self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # Rift特有: backcast用SiLU激活
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_act = nn.SiLU()
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, num_nodes, num_feat] = hidden_inherent_signal.shape
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        # Rift特有: SiLU激活的backcast
        backcast_seq = self.backcast_act(self.backcast_fc(hidden_states_inh))
        backcast_seq_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)
        if _RF_DBG:
            print(f"[RF-DBG:inh_block] rnn_out={hidden_states_rnn.shape} "
                  f"inh_out={hidden_states_inh.shape} fk={forecast_hidden.shape}",
                  file=sys.stderr, flush=True)
        return backcast_seq_res, forecast_hidden
