"""Flux InhBlock: 流式感知的inherent块.
与upstream(标准PE + RNN + Transformer)不同,
Flux使用指数衰减位置编码: 越远的位置编码强度越弱,
配合流式推理的因果性. 同时forecast分支使用渐进式置信度衰减."""
import math

import torch
import torch.nn as nn
import sys
import os

from walpurgis_flux.models.decouple.residual_decomp import \
    ResidualDecomp
from walpurgis_flux.models.inherent_block.inh_model import \
    RNNLayer, TransformerLayer
from walpurgis_flux.models.inherent_block.forecast import \
    Forecast

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class ExponentialDecayPE(nn.Module):
    """Flux特有: 指数衰减位置编码.
    标准PE对所有位置赋予等权重的sin/cos编码,
    而Flux的PE对远离当前位置的编码施加指数衰减.
    这使得模型在流式推理时更关注近期信息."""
    def __init__(self, d_model, dropout=None,
                 max_len=5000, decay_rate=0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.decay_rate = decay_rate
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # 可学习衰减速率
        self.learned_decay = nn.Parameter(
            torch.tensor(decay_rate))

    def forward(self, X):
        seq_len = X.size(0)
        # 生成指数衰减掩码: 最后位置权重=1, 往前衰减
        decay = torch.sigmoid(self.learned_decay)
        positions = torch.arange(
            seq_len, device=X.device).float()
        decay_mask = torch.exp(
            -decay * (seq_len - 1 - positions))
        decay_mask = decay_mask.view(-1, 1, 1)
        # 加入衰减的位置编码
        X = X + self.pe[:seq_len] * decay_mask
        X = self.dropout(X)
        if _FX_DBG:
            print(f"[FX:exp_decay_pe] seq={seq_len} "
                  f"decay={decay.item():.4f} "
                  f"mask_range=["
                  f"{decay_mask.min().item():.4f},"
                  f"{decay_mask.max().item():.4f}]",
                  file=sys.stderr)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,
                 bias=True, forecast_hidden_dim=256,
                 **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        # Flux: 指数衰减位置编码替代标准PE
        self.pos_encoder = ExponentialDecayPE(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(
            hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads,
            model_args['dropout'], bias)
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
        ## Flux: exponential decay PE
        hidden_states_rnn = self.pos_encoder(
            hidden_states_rnn)
        ## MSA with chunked causal attention
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn,
            hidden_states_rnn)
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
