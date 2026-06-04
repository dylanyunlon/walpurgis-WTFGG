import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from walpurgis import _dbg

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_TAG = "inhblk"


def _mish(x):
    return x * torch.tanh(F.softplus(x))


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # 改动1: 去掉 sincos PE, 用 forecast 内的 RoPE
        # upstream 有单独的 PositionalEncoding 模块
        # 在 Forecast.forward 里用 RoPE, 这里不重复加 PE
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = Forecast(
            hidden_dim, forecast_hidden_dim, **model_args)

        # 改动2: 单层 FC → 2层 MLP + Mish
        self.backcast_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

        # 改动3: 可学习残差门 — upstream 直接相加
        self.res_gate_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_inherent_signal):
        B, L, N, D = hidden_inherent_signal.shape

        _dbg(_TAG, "input", signal=hidden_inherent_signal)

        # RNN
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)

        # 改动1: 不加 PE — RoPE 在 forecast 内部处理
        # upstream: hidden_states_rnn = self.pos_encoder(hidden_states_rnn)

        # 改动4: gradient checkpoint 包裹 transformer
        if self.training:
            hidden_states_inh = checkpoint(
                self.transformer_layer,
                hidden_states_rnn, hidden_states_rnn, hidden_states_rnn,
                use_reentrant=False)
        else:
            hidden_states_inh = self.transformer_layer(
                hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        _dbg(_TAG, "transformer_done", inh=hidden_states_inh)

        # forecast — pe 参数设为 None, RoPE 在 forecast 内部
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn,
            hidden_states_inh, self.transformer_layer,
            self.rnn_layer, None)

        # 改动2: 2层 MLP backcast
        hidden_states_inh = hidden_states_inh.reshape(
            L, B, N, D).transpose(0, 1)
        backcast_seq = self.backcast_mlp(hidden_states_inh)

        # 改动3: sigmoid 门控残差
        gate = torch.sigmoid(self.res_gate_fc(hidden_states_inh))
        backcast_seq = backcast_seq * gate

        backcast_seq_res = self.residual_decompose(
            hidden_inherent_signal, backcast_seq)

        _dbg(_TAG, "output",
             gate_mean=gate.mean(), backcast=backcast_seq_res,
             forecast=forecast_hidden)

        return backcast_seq_res, forecast_hidden
