"""Nebula DifBlock: Highway network backcast + GroupNorm residual."""
import torch, torch.nn as nn, sys, os
from .forecast import Forecast; from .dif_model import STLocalizedConv; from ..decouple.residual_decomp import ResidualDecomp
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class HighwayBackcast(nn.Module):
    """Highway network for backcast: T(x)*H(x) + (1-T(x))*x.
    Replaces upstream plain Linear backcast with gated highway layer."""
    def __init__(self, dim):
        super().__init__()
        self.H = nn.Linear(dim, dim)  # transform
        self.T = nn.Linear(dim, dim)  # gate
        nn.init.constant_(self.T.bias, -1.0)  # initialize gate bias negative -> pass-through

    def forward(self, x):
        h = torch.tanh(self.H(x))
        t = torch.sigmoid(self.T(x))
        return t * h + (1.0 - t) * x


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # Nebula: highway network backcast replaces plain Linear
        self.backcast_branch = HighwayBackcast(hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        # Nebula: highway backcast
        backcast_seq = self.backcast_branch(hidden_states_dif)
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)
        if _NEB_DBG:
            print(f"[NEB:block@dif_block] backcast={list(backcast_seq.shape)} forecast={list(forecast_hidden.shape)}", file=sys.stderr)
        return backcast_seq_res, forecast_hidden
