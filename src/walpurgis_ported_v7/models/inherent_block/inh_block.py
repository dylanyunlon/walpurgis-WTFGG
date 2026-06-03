import math
import torch
import torch.nn as nn
import sys

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_DBG_INHBLK = ("--dbg-inhblk" in sys.argv)


class PositionalEncoding(nn.Module):
    """算法改动: hybrid PE — 固定 sinusoidal + 可学习偏移
    原版纯 sinusoidal; 加一个可学习的 pe_bias 让模型自适应微调位置信息
    """
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # 可学习偏移 (最多 max_len 个)
        self.pe_bias = nn.Parameter(torch.zeros(max_len, 1, d_model) * 0.01)

    def forward(self, X):
        seq_len = X.size(0)
        X = X + self.pe[:seq_len] + self.pe_bias[:seq_len]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = PositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)

        # 算法改动: backcast 分支加 dropout 防过拟合
        self.backcast_drop = nn.Dropout(model_args['dropout'])

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, num_nodes, num_feat] = \
            hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        if _DBG_INHBLK:
            with torch.no_grad():
                print(f"[DBG-INHBLK] rnn_out_norm="
                      f"{hidden_states_rnn.norm().item():.4f}  "
                      f"tf_out_norm={hidden_states_inh.norm().item():.4f}")

        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)

        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)

        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq = self.backcast_drop(backcast_seq)  # 算法改动: dropout

        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        if _DBG_INHBLK:
            with torch.no_grad():
                print(f"[DBG-INHBLK] backcast_norm="
                      f"{backcast_seq.norm().item():.4f}  "
                      f"residual_norm={backcast_seq_res.norm().item():.4f}")

        return backcast_seq_res, forecast_hidden
