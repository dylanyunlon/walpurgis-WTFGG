"""
DifBlock — walpurgis_ported_v4
Modifications:
  - forward: added input/output norm ratio tracking for gradient health monitoring
  - Uses v4 versions of STLocalizedConv, Forecast, ResidualDecomp
"""
import torch.nn as nn
import sys

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_V4_DEBUG = True


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None,
                 dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        # diffusion model
        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args)

        # forecast
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        input_norm = history_data.detach().norm().item() if _V4_DEBUG else 0

        # diffusion model
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        # forecast branch
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        # backcast branch
        backcast_seq = self.backcast_branch(hidden_states_dif)
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        if _V4_DEBUG:
            out_norm = backcast_seq_res.detach().norm().item()
            fk_norm = forecast_hidden.detach().norm().item()
            print(f"[v4-DBG][DifBlock] "
                  f"||input||={input_norm:.4f} ||backcast_res||={out_norm:.4f} "
                  f"||forecast||={fk_norm:.4f} "
                  f"ratio={out_norm/(input_norm+1e-8):.4f}",
                  file=sys.stderr)

        return backcast_seq_res, forecast_hidden
