import math
import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast

_TAG = "inh_block"


class ExponentialDecayPE(nn.Module):
    """upstream: sinusoidal positional encoding (PE)
    改动: 指数衰减PE — 越近的时间步权重越大
    traffic forecasting中近期数据比远期重要
    PE(t) = exp(-alpha * (T-t)) * embed, alpha可学习
    """

    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可学习的衰减率
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        # 基础embedding
        self.embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, X):
        seq_len = X.size(0)
        # 指数衰减: 最后一步=1, 越早越小
        positions = torch.arange(seq_len, device=X.device).float()
        decay = torch.exp(
            -self.decay_rate.abs() * (seq_len - 1 - positions))
        # [seq_len, 1, 1]
        decay = decay.unsqueeze(-1).unsqueeze(-1)
        X = X + decay * self.embed
        X = self.dropout(X)
        _dbg(f"{_TAG}/exp_pe_decay_rate",
             self.decay_rate, _TAG)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # 改动: 用ExponentialDecayPE代替sinusoidal PE
        self.pos_encoder = ExponentialDecayPE(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(
            hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = \
            hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        _dbg(f"{_TAG}/inh_attn_out", hidden_states_inh, _TAG)

        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, self.pos_encoder)

        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        return backcast_seq_res, forecast_hidden
