"""
InhBlock — Transit变体
使用S4 (Structured State Space) + GatedSelfAttention, 带诊断
"""
import math
import torch
import torch.nn as nn
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import S4Layer, GatedSelfAttention
from .forecast import Forecast
from ... import _dbg, dataflow_checkpoint


class PositionalEncoding(nn.Module):
    """正弦位置编码 + 可学习缩放因子"""

    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        scale = torch.clamp(self.pe_scale, min=0.01, max=5.0)
        X = X + scale * self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # S4 Structured State Space 替代 GRU
        self.pos_encoder = PositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.s4_layer = S4Layer(
            hidden_dim, state_dim=min(16, hidden_dim),
            dropout=model_args['dropout'])
        # 门控自注意力替代标准self-attention
        self.transformer_layer = GatedSelfAttention(
            hidden_dim, num_heads,
            model_args['dropout'], bias)
        # forecast / backcast
        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = \
            hidden_inherent_signal.shape

        dataflow_checkpoint(
            "inh_block.input", hidden_inherent_signal)

        # 重塑为S4输入: [seq_len, batch*nodes, feat]
        X_reshaped = hidden_inherent_signal.transpose(0, 1).reshape(
            seq_len, batch_size * num_nodes, num_feat)

        # S4序列建模
        hidden_states_s4 = self.s4_layer(X_reshaped)
        # 位置编码
        hidden_states_s4 = self.pos_encoder(hidden_states_s4)

        # 门控自注意力
        hidden_states_inh = self.transformer_layer(
            hidden_states_s4)

        _dbg("inh_block.s4_out",
             hidden_states_s4, "inherent")
        _dbg("inh_block.attn_out",
             hidden_states_inh, "inherent")

        # forecast
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_s4,
            hidden_states_inh, self.transformer_layer,
            self.s4_layer, self.pos_encoder)
        # backcast
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        return backcast_seq_res, forecast_hidden
