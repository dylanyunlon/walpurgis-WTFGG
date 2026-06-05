"""
DifBlock — Nightfall变体
算法改写:
  1. backcast分支加可学习缩放因子 (控制backcast力度)
  2. residual decompose前加dropout (正则化)
"""
import torch.nn as nn
from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp
from ... import _dbg


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
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        # backcast缩放 + residual前dropout
        self.backcast_scale = nn.Parameter(__import__('torch').tensor(1.0))
        self.resid_dropout = nn.Dropout(model_args.get('dropout', 0.3) * 0.3)

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)
        backcast_seq = self.backcast_branch(hidden_states_dif)
        # 可学习backcast缩放
        backcast_seq = backcast_seq * self.backcast_scale
        _dbg("dif_block.backcast_scale", self.backcast_scale, "model")
        backcast_seq = self.resid_dropout(backcast_seq)
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)
        return backcast_seq_res, forecast_hidden
