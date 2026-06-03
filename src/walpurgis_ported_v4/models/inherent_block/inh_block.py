import math
import torch
import torch.nn as nn
import sys

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_V4_DEBUG = True
_dbg_call_count = 0


def _dbg(tag, **kw):
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][InhBlock][{tag}]"]
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}|norm={v.detach().norm().item():.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" ".join(parts), file=sys.stderr)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # v4: use log-spaced frequencies with learnable phase shift
        # instead of fixed sin/cos, learn an additive phase offset per dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # v4: learnable phase offset — lets the model adapt PE to dataset periodicity
        self.phase_offset = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, X):
        # v4: add learnable phase to static sinusoidal PE
        X = X + self.pe[:X.size(0)] + self.phase_offset
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # inherent model
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)

        # forecast branch
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # backcast branch
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)

        # v4: gated residual for backcast — learnable interpolation between
        # backcast output and skip connection, replacing hard residual decomp
        self.backcast_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        global _dbg_call_count
        _dbg_call_count += 1

        [batch_size, seq_len, num_nodes, num_feat] = hidden_inherent_signal.shape

        # inherent model
        ## rnn
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        ## pe
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        ## MSA
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # forecast branch
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder
        )

        # backcast branch
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)

        # v4: gated residual — learn how much of the backcast to keep vs skip
        gate_input = torch.cat([backcast_seq, hidden_inherent_signal], dim=-1)
        gate = self.backcast_gate(gate_input)
        backcast_seq = gate * backcast_seq + (1 - gate) * hidden_inherent_signal

        backcast_seq_res = self.residual_decompose(hidden_inherent_signal, backcast_seq)

        if _V4_DEBUG and _dbg_call_count <= 5:
            _dbg("forward",
                 input=hidden_inherent_signal,
                 rnn_out=hidden_states_rnn,
                 backcast=backcast_seq,
                 gate_mean=f"{gate.detach().mean().item():.4f}",
                 forecast=forecast_hidden)

        return backcast_seq_res, forecast_hidden
