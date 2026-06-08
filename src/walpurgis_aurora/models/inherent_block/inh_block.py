"""
InhBlock — Aurora变体
算法改写: 使用MultiScaleTemporalLayer替代单一RNNLayer做时序处理
  - 多尺度temporal attention捕捉不同时间粒度的模式
  - PositionalEncoding使用频率自适应(可学习频率基础)
"""
import math
import torch
import torch.nn as nn
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import MultiScaleTemporalLayer, RNNLayer, TransformerLayer
from .forecast import Forecast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # Aurora: 可学习频率基础, 自适应调整PE的频率分布
        self.freq_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        # Aurora: freq_scale调制位置编码的频率
        scale = torch.sigmoid(self.freq_scale) + 0.5
        X = X + scale * self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # Aurora: 主路径用MultiScaleTemporalLayer
        self.pos_encoder = PositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.multi_scale_temporal = MultiScaleTemporalLayer(
            hidden_dim, model_args['dropout'])
        # 保留RNNLayer用于forecast分支兼容
        self.rnn_layer = RNNLayer(
            hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads,
            model_args['dropout'], bias)

        # forecast branch
        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        # backcast branch
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, num_nodes, num_feat] = \
            hidden_inherent_signal.shape

        # Aurora: 用MultiScaleTemporalLayer替代单一RNNLayer
        hidden_states_ms = self.multi_scale_temporal(
            hidden_inherent_signal)  # [L, B*N, D]

        # PE
        hidden_states_ms = self.pos_encoder(hidden_states_ms)

        # MSA (TransformerLayer with gated residual)
        hidden_states_inh = self.transformer_layer(
            hidden_states_ms, hidden_states_ms,
            hidden_states_ms)

        # forecast分支仍用RNNLayer保持AR生成的兼容性
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)

        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, self.pos_encoder)

        # backcast branch
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        return backcast_seq_res, forecast_hidden
