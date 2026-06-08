"""
InhBlock — Parallax变体 (M054)
使用xLSTM + Positional Interpolation + CrossAttention, 带诊断

算法改动 vs Penumbra:
  - MinGRU → xLSTM: 指数门控+归一化器+记忆混合
  - 正弦位置编码 → Positional Interpolation: 支持序列外推
  - 增加head_dim可配置参数, 用于xLSTM多头扩展
"""
import math
import torch
import torch.nn as nn
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import (XLSTMLayer, PositionalInterpolation,
                        CrossAttentionLayer)
from .forecast import Forecast
from ... import _dbg, dataflow_checkpoint


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # xLSTM替代MinGRU — 指数门控+归一化器
        self.pos_encoder = PositionalInterpolation(
            hidden_dim, model_args['dropout'],
            train_len=model_args.get('seq_length', 12))
        self.rnn_layer = XLSTMLayer(
            hidden_dim, model_args['dropout'])
        # CrossAttention: 门控self/cross混合
        self.transformer_layer = CrossAttentionLayer(
            hidden_dim, num_heads,
            model_args['dropout'], bias)
        # forecast / backcast
        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

        print(f"[PAR-DBG] InhBlock: xLSTM(dim={hidden_dim}) "
              f"+ PosInterp(train_len="
              f"{model_args.get('seq_length', 12)}) "
              f"+ CrossAttn(heads={num_heads})")

    def forward(self, hidden_inherent_signal):
        batch_size, seq_len, num_nodes, num_feat = \
            hidden_inherent_signal.shape

        dataflow_checkpoint(
            "inh_block.input", hidden_inherent_signal)

        # xLSTM: 指数门控 + 归一化器状态
        hidden_states_rnn = self.rnn_layer(
            hidden_inherent_signal)
        # 位置插值编码 — 支持外推
        hidden_states_rnn = self.pos_encoder(
            hidden_states_rnn)

        _dbg("inh_block.xlstm_out",
             hidden_states_rnn, "inherent")
        _dbg("inh_block.xlstm_range",
             f"[{hidden_states_rnn.min().item():.4f}, "
             f"{hidden_states_rnn.max().item():.4f}]",
             "inherent")

        # Cross-Attention: Q=xLSTM输出, 同时cross K/V=原始信号
        original_signal = hidden_inherent_signal.transpose(
            0, 1).reshape(
            seq_len, batch_size * num_nodes, num_feat)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn,
            hidden_states_rnn,
            cross_K=original_signal,
            cross_V=original_signal)

        _dbg("inh_block.attn_out",
             hidden_states_inh, "inherent")

        # forecast
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, self.pos_encoder)
        # backcast
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        return backcast_seq_res, forecast_hidden
