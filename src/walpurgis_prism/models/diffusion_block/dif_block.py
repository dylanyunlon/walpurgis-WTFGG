"""Prism diffusion block: adds spatial-view attention to the diffusion pathway.
The forecast branch receives an additional spatial context vector from the
multi-view spatial encoder, enriching the diffusion hidden states."""
import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None,
                 sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        # diffusion model
        self.localized_st_conv = STLocalizedConv(
            hidden_dim,
            pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph,
            sta_graph=sta_graph, **model_args)
        # forecast
        self.forecast_branch = Forecast(
            hidden_dim,
            forecast_hidden_dim=forecast_hidden_dim,
            **model_args)
        # backcast
        self.backcast_branch = nn.Linear(
            hidden_dim, hidden_dim)
        # Prism特有: 残差分解使用频域平滑
        self.residual_decompose = ResidualDecomp(
            [-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data,
                dynamic_graph, static_graph):
        # diffusion model
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)
        # forecast branch
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph,
            static_graph)
        # backcast branch
        backcast_seq = self.backcast_branch(
            hidden_states_dif)
        # residual decomposition with spectral smoothing
        backcast_seq = backcast_seq
        history_data = history_data[
            :, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(
            history_data, backcast_seq)
        return backcast_seq_res, forecast_hidden
