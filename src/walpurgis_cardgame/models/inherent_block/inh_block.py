"""
inh_block.py — CardGame InhBlock
算法改写 (vs upstream):
  - 直接减法残差 → tanh门控gated residual
  - backcast = gate * hidden + (1-gate) * input
  - PositionalEncoding保持不变
"""
import os
import sys
import math
import torch
import torch.nn as nn

from walpurgis_cardgame.models.decouple.residual_decomp import ResidualDecomp
from walpurgis_cardgame.models.inherent_block.inh_model import RNNLayer, TransformerLayer
from walpurgis_cardgame.models.inherent_block.forecast import Forecast

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="InhBlock"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class PositionalEncoding(nn.Module):
    """标准正弦余弦位置编码"""

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

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class GatedResidual(nn.Module):
    """CardGame: tanh门控残差
    output = tanh(gate) * transformed + (1 - tanh(gate)) * original
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, original, transformed):
        combined = torch.cat([original, transformed], dim=-1)
        gate = torch.tanh(self.gate_fc(combined))
        return gate * transformed + (1 - gate) * original


class InhBlock(nn.Module):
    """CardGame Inherent Block with tanh gated residual"""

    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # inherent model
        self.pos_encoder = PositionalEncoding(
            hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        # forecast branch
        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)

        # CardGame: tanh gated residual替代简单减法
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gated_residual = GatedResidual(hidden_dim)

        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        _dbg("input", hidden_inherent_signal)

        [batch_size, seq_len, num_nodes, num_feat] = \
            hidden_inherent_signal.shape

        # inherent model
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # forecast branch
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder)

        # backcast branch with tanh gated residual
        hidden_states_inh = hidden_states_inh.reshape(
            seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)

        # CardGame: tanh gated residual
        backcast_seq_res = self.gated_residual(
            hidden_inherent_signal, backcast_seq)
        _dbg("gated_residual_output", backcast_seq_res)

        return backcast_seq_res, forecast_hidden
