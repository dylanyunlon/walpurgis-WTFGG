import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

# Delta vs upstream:
#   1. Backcast branch: plain Linear → Linear + GELU + Linear (2-layer)
#   2. Added pre-LayerNorm on hidden_states before forecast branch

class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args)

        # forecast
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)

        # ── delta 1: 2-layer backcast ──
        self.backcast_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── delta 2: pre-LN before forecast ──
        self.pre_ln = nn.LayerNorm(hidden_dim)

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        # ── delta 2: normalise before forecast branch ──
        normed = self.pre_ln(hidden_states_dif)
        forecast_hidden = self.forecast_branch(
            gated_history_data, normed,
            self.localized_st_conv, dynamic_graph, static_graph)

        # ── delta 1: deeper backcast ──
        backcast_seq = self.backcast_branch(hidden_states_dif)
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden
