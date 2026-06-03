"""Diffusion block — with feature-attention gating.

Changes
-------
Adds a lightweight 1-layer feature-attention gate between the backcast
branch output and the residual decomposition.  The gate learns a
per-feature weighting of the backcast signal, allowing the model to
selectively suppress noisy spatial dimensions before subtraction.
"""

import torch
import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp
from walpurgis_ported_v6 import _dbg


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None,
                 **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args)

        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim,
            **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)

        # ── feature-attention gate on backcast ──
        self.feat_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data,
                dynamic_graph, static_graph):
        hidden_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        backcast_seq = self.backcast_branch(hidden_dif)
        # apply feature attention
        gate_weight = self.feat_gate(backcast_seq)
        backcast_seq = backcast_seq * gate_weight

        _dbg("DifBlock.gate", gate_weight)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden
