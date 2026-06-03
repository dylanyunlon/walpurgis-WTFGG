"""
dif_block.py — v9 port
Algo delta:
  1. backcast_branch: 单 FC → 两层 MLP + SiLU,
     给 backcast 更强的表达能力
  2. backcast 输出后加 feature-attention gate:
     channel_gate = σ(FC(mean_pool(backcast))), backcast *= gate
     类似 SE-Net 的通道注意力, 让网络学习抑制哪些特征维度
"""
import torch
import torch.nn as nn
from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp
from walpurgis_ported_v9 import _dbg

_TAG = "dif_blk"


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)

        self.forecast_branch = Forecast(hidden_dim,
                                        forecast_hidden_dim=forecast_hidden_dim, **model_args)

        # v9: two-layer MLP backcast
        self.backcast_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_act = nn.SiLU()
        self.backcast_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # v9: channel attention gate (SE-Net style)
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        # v9: MLP backcast
        bc = self.backcast_fc1(hidden_states_dif)
        bc = self.backcast_act(bc)
        bc = self.backcast_fc2(bc)

        # v9: channel attention gate
        pooled = bc.mean(dim=(1, 2))                     # [B, D]
        gate = self.channel_gate(pooled).unsqueeze(1).unsqueeze(2)  # [B,1,1,D]
        bc = bc * gate

        _dbg(_TAG, f"channel_gate∈[{gate.min().item():.3f},{gate.max().item():.3f}]  "
                    f"bc_std={bc.std().item():.6g}")

        history_data = history_data[:, -bc.shape[1]:, :, :]
        backcast_res = self.residual_decompose(history_data, bc)
        return backcast_res, forecast_hidden
