import torch
import torch.nn as nn
from walpurgis import _dbg

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_TAG = "difblk"


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args)

        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)

        # 改动1: 单层 FC → 3层 MLP + GELU
        # upstream: nn.Linear(hidden, hidden) 单层 backcast
        # walpurgis改动: 3层 MLP 给 backcast 更多表达能力
        self.backcast_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 改动2: 门控残差 — 可学习 sigmoid gate 控制 backcast 强度
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data,
                dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        _dbg(_TAG, "st_conv_out", hidden=hidden_states_dif)

        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        # 改动1: 3层 MLP backcast
        backcast_seq = self.backcast_mlp(hidden_states_dif)

        # 改动2: sigmoid 门控 — upstream 无此门
        gate = torch.sigmoid(self.gate_fc(hidden_states_dif))
        backcast_seq = backcast_seq * gate

        _dbg(_TAG, "backcast", gate_mean=gate.mean(), backcast=backcast_seq)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        # 改动3: skip shortcut — upstream 纯 residual decompose
        # 加一个从 history 到输出的 shortcut, 防梯度消失
        backcast_seq_res = backcast_seq_res + 0.1 * history_data

        _dbg(_TAG, "residual_out", res=backcast_seq_res)
        return backcast_seq_res, forecast_hidden
