"""Prism inherent block: adds cross-view temporal attention.
After RNN + Transformer processing, a cross-attention layer attends between
the temporal view (RNN output) and spatial view (Transformer output),
enriching the inherent representation with multi-view information."""
import math
import torch
import torch.nn as nn

from walpurgis_prism.models.decouple.residual_decomp import ResidualDecomp
from walpurgis_prism.models.inherent_block.inh_model import RNNLayer, TransformerLayer
from walpurgis_prism.models.inherent_block.forecast import Forecast


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

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class CrossViewAttention(nn.Module):
    """Prism特有: 时间-空间交叉视角注意力
    用RNN输出(时间视角)作为Query, Transformer输出(空间视角)作为Key/Value,
    通过交叉注意力融合两个视角的信息。"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.blend_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, temporal_feat, spatial_feat):
        # temporal_feat: [L, B*N, D] (from RNN)
        # spatial_feat: [L, B*N, D] (from Transformer)
        cross_out, _ = self.cross_attn(
            temporal_feat, spatial_feat, spatial_feat)
        gate = torch.sigmoid(self.blend_gate)
        fused = gate * cross_out + (1 - gate) * temporal_feat
        fused = self.norm(fused)
        return fused


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,
                 bias=True, forecast_hidden_dim=256,
                 **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # inherent model
        self.pos_encoder = PositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(
            hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads,
            model_args['dropout'], bias)
        # Prism特有: 交叉视角注意力
        self.cross_view_attn = CrossViewAttention(
            hidden_dim, num_heads,
            model_args['dropout'])
        # forecast branch
        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim,
            **model_args)
        # backcast branch
        self.backcast_fc = nn.Linear(
            hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, num_nodes,
         num_feat] = hidden_inherent_signal.shape
        # inherent model
        ## rnn
        hidden_states_rnn = self.rnn_layer(
            hidden_inherent_signal)
        ## pe
        hidden_states_rnn = self.pos_encoder(
            hidden_states_rnn)
        ## MSA
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn,
            hidden_states_rnn)
        # Prism特有: 交叉视角注意力融合RNN和Transformer输出
        hidden_states_inh = self.cross_view_attn(
            hidden_states_rnn, hidden_states_inh)
        # forecast branch
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, self.pos_encoder)
        # backcast branch
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(
            0, 1)
        backcast_seq = self.backcast_fc(
            hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)
        return backcast_seq_res, forecast_hidden
