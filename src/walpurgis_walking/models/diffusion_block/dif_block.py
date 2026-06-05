import torch
import torch.nn as nn
from walpurgis_walking import _dbg

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_TAG = "dif_blk"


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)

        # 改动1: 3-layer MLP backcast + GELU (upstream 单层 Linear)
        self.backcast_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # 改动2: 残差门控 (upstream 无)
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_dif, self.localized_st_conv,
            dynamic_graph, static_graph)

        # 改动: 3-layer MLP + 门控
        bc_raw = self.backcast_branch(hidden_dif)
        gate = torch.sigmoid(self.gate_fc(hidden_dif))
        backcast_seq = bc_raw * gate

        _dbg(_TAG, "gate", gate_mean=gate.mean(), bc_norm=bc_raw.norm())

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        _dbg(_TAG, "out", res_norm=backcast_seq_res.norm(), fk_norm=forecast_hidden.norm())
        return backcast_seq_res, forecast_hidden
