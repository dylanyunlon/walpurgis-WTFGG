"""
InhBlock — Umbra变体
算法改动: Mamba SSM + ALiBi位置偏差
  原版(Penumbra): MinGRU + CrossAttention + 正弦位置编码
  Umbra:
    - MambaSSMLayer替代MinGRU: 选择性状态空间模型,
      输入依赖的B(x)/C(x)矩阵 + ZOH离散化
    - ALiBiAttention替代标准MultiheadAttention:
      不使用正弦位置编码, 直接给attention score加线性距离惩罚
      score_ij -= slope * |i - j|
    - 去掉PositionalEncoding, ALiBi已包含位置信息
"""
import torch
import torch.nn as nn
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import MambaSSMLayer, ALiBiAttentionLayer
from .forecast import Forecast
from ... import _dbg, dataflow_checkpoint


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # Mamba SSM替代MinGRU — 无需PositionalEncoding
        self.rnn_layer = MambaSSMLayer(
            hidden_dim, dropout=model_args['dropout'])
        # ALiBi注意力替代标准自注意力 — 位置信息由线性偏差提供
        self.transformer_layer = ALiBiAttentionLayer(
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

        # Mamba SSM: 选择性状态空间扫描
        hidden_states_rnn = self.rnn_layer(
            hidden_inherent_signal)

        _dbg("inh_block.mamba_out",
             hidden_states_rnn, "inherent")
        _dbg("inh_block.mamba_out_stats",
             f"mean={hidden_states_rnn.mean().item():.6f} "
             f"std={hidden_states_rnn.std().item():.6f} "
             f"shape={list(hidden_states_rnn.shape)}",
             "inherent")

        # 注意: 不使用PositionalEncoding
        # ALiBi注意力自带位置偏差, 直接在score上
        # 加 -slope * |i-j|, 无需显式PE

        # ALiBi注意力: Q=Mamba输出, K/V也是Mamba输出
        # 同时提供原始信号作为交叉注意力的K/V
        original_signal = hidden_inherent_signal.transpose(
            0, 1).reshape(
            seq_len, batch_size * num_nodes, num_feat)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn,
            hidden_states_rnn,
            cross_K=original_signal,
            cross_V=original_signal)

        _dbg("inh_block.alibi_attn_out",
             hidden_states_inh, "inherent")
        _dbg("inh_block.alibi_attn_out_range",
             f"[{hidden_states_inh.min().item():.4f}, "
             f"{hidden_states_inh.max().item():.4f}]",
             "inherent")

        # forecast
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer)
        # backcast
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        _dbg("inh_block.backcast_residual_norm",
             f"{backcast_seq_res.norm().item():.4f}",
             "inherent")

        return backcast_seq_res, forecast_hidden
