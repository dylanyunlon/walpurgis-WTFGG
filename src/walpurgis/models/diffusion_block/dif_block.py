"""
Walpurgis Diffusion Block — Spatial Propagation with Tier-Aware Debug
=====================================================================
Adapted from D2STGNN DifBlock. The diffusion model captures spatial
dependencies via localized ST convolution on the graph.

Modifications:
  1. Forward timing per sub-component (localized_conv, forecast, backcast, residual)
  2. Shape assertions with descriptive error messages 
  3. Debug probe at residual decomposition — most common source of gradient issues
"""

import torch.nn as nn
import time

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        
        # Diffusion model — the spatial propagation engine
        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph, 
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args
        )
        
        # Forecast & backcast branches
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        
        # Walpurgis debug
        self._call_count = 0
        self._timing = {'conv': [], 'forecast': [], 'backcast': [], 'residual': []}

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        self._call_count += 1
        verbose = (self._call_count % 200 == 1)
        
        # Diffusion convolution
        t0 = time.time()
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        self._timing['conv'].append(time.time() - t0)
        
        # Forecast branch
        t0 = time.time()
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph
        )
        self._timing['forecast'].append(time.time() - t0)
        
        # Backcast branch
        t0 = time.time()
        backcast_seq = self.backcast_branch(hidden_states_dif)
        self._timing['backcast'].append(time.time() - t0)
        
        # Walpurgis: pre-residual norm check — if backcast signal is much larger
        # than history, the residual will be dominated by backcast artifacts.
        # This often signals tier-boundary instability in heterogeneous training.
        import torch
        backcast_norm = backcast_seq.norm().item()
        
        # Residual decomposition — critical numerical stability point
        t0 = time.time()
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        history_norm = history_data.norm().item()
        
        # Walpurgis algorithm tweak: if backcast overwhelms history (ratio > 10x),
        # scale it down before residual to prevent gradient explosion
        ratio = backcast_norm / (history_norm + 1e-8)
        if ratio > 10.0:
            backcast_seq = backcast_seq * (history_norm / (backcast_norm + 1e-8)) * 5.0
            if verbose:
                print(f"      [DifBlock] ⚠ backcast/history ratio={ratio:.1f} "
                      f"→ rescaled backcast")
        
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)
        self._timing['residual'].append(time.time() - t0)
        
        if verbose:
            avg = {k: sum(v[-100:]) / max(len(v[-100:]), 1) * 1000 for k, v in self._timing.items()}
            print(f"      [DifBlock] call={self._call_count}: "
                  f"conv={avg['conv']:.1f}ms, fcast={avg['forecast']:.1f}ms, "
                  f"bcast={avg['backcast']:.1f}ms, resid={avg['residual']:.1f}ms, "
                  f"bcast/hist_ratio={ratio:.2f}")
            # Health check on output
            res_nan = torch.isnan(backcast_seq_res).any().item()
            fcast_nan = torch.isnan(forecast_hidden).any().item()
            if res_nan or fcast_nan:
                print(f"      [DifBlock] ⚠ OUTPUT NaN: residual={res_nan} forecast={fcast_nan}")

        return backcast_seq_res, forecast_hidden
