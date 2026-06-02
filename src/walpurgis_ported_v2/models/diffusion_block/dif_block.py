"""
Diffusion Block: wraps ST-localized convolution with forecast / backcast
branches and residual decomposition.
"""

import torch.nn as nn
import sys

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_DBG_DIFBLK = ("--debug-difblk" in sys.argv) or False


class DifBlock(nn.Module):
    """
    Contains:
      1. STLocalizedConv — diffusion on gated input
      2. Forecast branch — AR roll-out for future prediction
      3. Backcast branch — FC projection for residual subtraction
      4. ResidualDecomp  — strip learned signal from original
    """

    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.predef = model_args['adjs']

        # core diffusion conv
        self.st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.predef,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args
        )

        # branches
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args
        )
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_link   = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history, gated_history, dynamic_graph, static_graph):
        """
        Parameters
        ----------
        history        : [B, L, N, D]  — raw (un-gated) hidden states
        gated_history  : [B, L, N, D]  — estimation-gate output
        dynamic_graph  : list[Tensor]
        static_graph   : list[Tensor]

        Returns
        -------
        backcast_residual : [B, L', N, D]  — feed to inherent block
        forecast_hidden   : [B, H, N, fk_dim] — future prediction features
        """
        # diffusion on gated signal
        dif_hidden = self.st_conv(gated_history, dynamic_graph, static_graph)

        # forecast: AR roll-out using same conv
        fk_hidden = self.forecast_branch(
            gated_history, dif_hidden, self.st_conv, dynamic_graph, static_graph
        )

        # backcast: simple projection
        bc_seq = self.backcast_branch(dif_hidden)

        # residual: strip backcast from original (aligned temporally)
        history_aligned = history[:, -bc_seq.shape[1]:, :, :]
        residual_out = self.residual_link(history_aligned, bc_seq)

        if _DBG_DIFBLK:
            print(f"[DBG:difblk] dif_hidden={tuple(dif_hidden.shape)}  "
                  f"fk={tuple(fk_hidden.shape)}  "
                  f"residual={tuple(residual_out.shape)}  "
                  f"dif_norm={dif_hidden.norm().item():.4f}")
        return residual_out, fk_hidden
