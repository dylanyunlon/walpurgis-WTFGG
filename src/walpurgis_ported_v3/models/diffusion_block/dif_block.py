"""
Diffusion Block: ST conv + forecast/backcast branches + residual decomp.
"""
import sys
import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_DBG = ("--debug-difblk" in sys.argv)


class DifBlock(nn.Module):

    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **kw):
        super().__init__()
        pre_g = kw['adjs']

        self.st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=pre_g,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **kw)

        self.fcast = Forecast(hidden_dim,
                              forecast_hidden_dim=forecast_hidden_dim, **kw)
        self.bcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.res_decomp = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_data, dynamic_graph, static_graph):
        diff_h = self.st_conv(gated_data, dynamic_graph, static_graph)

        # forecast path
        fk = self.fcast(gated_data, diff_h, self.st_conv,
                        dynamic_graph, static_graph)
        # backcast path
        bk = self.bcast_fc(diff_h)
        trimmed = history_data[:, -bk.shape[1]:, :, :]
        residual = self.res_decomp(trimmed, bk)

        if _DBG:
            print(f"[DBG:difblk] diff_h={tuple(diff_h.shape)}  "
                  f"fk={tuple(fk.shape)}  residual={tuple(residual.shape)}")

        return residual, fk
