"""Flux DifBlock: 流式感知扩散块.
与upstream(直接backcast + residual)和vortex(同upstream)不同,
Flux在backcast分支加入因果门控: 用sigmoid gate控制backcast信号的
通过比例, 使得流式推理时能够自适应调节残差信号强度."""
import torch.nn as nn
import torch
import sys
import os

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


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
        # Flux: 因果门控 — 控制backcast通过比例
        self.backcast_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid())
        # residual decomposition
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
        # backcast branch with causal gate
        backcast_seq = self.backcast_branch(
            hidden_states_dif)
        # Flux: 因果门控调节backcast强度
        gate_val = self.backcast_gate(hidden_states_dif)
        backcast_seq = backcast_seq * gate_val
        # residual decomposition
        history_data = history_data[
            :, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(
            history_data, backcast_seq)
        if _FX_DBG:
            print(f"[FX:dif_block] gate_mean="
                  f"{gate_val.mean().item():.4f} "
                  f"backcast_norm="
                  f"{backcast_seq.norm().item():.4f}",
                  file=sys.stderr)
        return backcast_seq_res, forecast_hidden
