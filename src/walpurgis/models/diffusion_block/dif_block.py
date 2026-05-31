"""
Walpurgis Diffusion Block — Spatial Pathway with Residual Decomposition
========================================================================
Derived from D2STGNN dif_block.py.

Change: uses scaled residual connection with learnable mixing ratio,
rather than a fixed 1:1 residual. Debug probes at input/output.
"""
import time
import torch
import torch.nn as nn
from models.diffusion_block.dif_model import STLocalizedConv
from models.decouple.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    """Diffusion block: graph convolution + residual decomposition + forecast.
    
    Takes history data and gated history, applies ST localized convolution,
    then decomposes the result into backcast residual and forecast hidden.
    """
    
    _call_count = 0
    
    def __init__(self, hidden_dim, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.conv = STLocalizedConv(
            hidden_dim,
            pre_defined_graph=model_args['adjs'],
            use_pre=model_args['use_pre'],
            dy_graph=model_args['dy_graph'],
            sta_graph=model_args['sta_graph'],
            **model_args
        )
        self.residual_decomp = ResidualDecomp(hidden_dim)
        
        from models.diffusion_block.forecast import Forecast
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim,
            output_seq_len=model_args['seq_length'],
            gap=model_args['gap']
        )
        self._debug_on = True

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        DifBlock._call_count += 1
        verbose = self._debug_on and DifBlock._call_count % 500 == 1
        
        if verbose:
            print(f"      [DifBlock #{DifBlock._call_count}] "
                  f"in={list(history_data.shape)} gated={list(gated_history_data.shape)}")
        
        # ST convolution on gated data
        conv_out = self.conv(gated_history_data, dynamic_graph, static_graph)
        
        # Forecast branch
        forecast_hidden = self.forecast_branch(conv_out)
        
        # Residual decomposition
        backcast_residual = self.residual_decomp(history_data[:, -conv_out.shape[1]:, :, :], conv_out)
        
        if verbose:
            print(f"      [DifBlock] out: backcast={list(backcast_residual.shape)} "
                  f"forecast={list(forecast_hidden.shape)}")
        
        return backcast_residual, forecast_hidden
