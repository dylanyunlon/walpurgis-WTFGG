"""
Walpurgis v2 Diffusion Block
==============================
Delta: adds pre-convolution dropout on the gated input (p=0.05) and
tracks input/output norm ratio for diagnosing gradient flow issues.
"""
import time
import torch
import torch.nn as nn
from models.diffusion_block.dif_model import STLocalizedConv
from models.decouple.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    _n = 0

    def __init__(self, hidden_dim, forecast_hidden_dim=256, **kw):
        super().__init__()
        self.conv = STLocalizedConv(
            hidden_dim,
            pre_defined_graph=kw["adjs"],
            use_pre=kw["use_pre"],
            dy_graph=kw["dy_graph"],
            sta_graph=kw["sta_graph"],
            **kw,
        )
        self.residual_decomp = ResidualDecomp(hidden_dim)
        from models.diffusion_block.forecast import Forecast
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim,
            output_seq_len=kw["seq_length"], gap=kw["gap"],
        )
        self._pre_drop = nn.Dropout(0.05)
        self._debug = True

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        DifBlock._n += 1
        verbose = self._debug and DifBlock._n % 500 == 1
        if verbose:
            print(
                f"      [DifBlock #{DifBlock._n}] in={list(history_data.shape)} "
                f"gated={list(gated_history_data.shape)}"
            )

        gated_drop = self._pre_drop(gated_history_data)
        conv_out = self.conv(gated_drop, dynamic_graph, static_graph)
        forecast_h = self.forecast_branch(conv_out)
        backcast_r = self.residual_decomp(
            history_data[:, -conv_out.shape[1]:], conv_out,
        )

        if verbose:
            ratio = conv_out.norm().item() / (gated_history_data.norm().item() + 1e-8)
            print(
                f"      [DifBlock] out: back={list(backcast_r.shape)} "
                f"fc={list(forecast_h.shape)} norm_ratio={ratio:.4f}"
            )
        return backcast_r, forecast_h
