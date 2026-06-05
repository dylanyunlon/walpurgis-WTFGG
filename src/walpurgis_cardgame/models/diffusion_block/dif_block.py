"""
dif_block.py — CardGame DifBlock
算法改写 (vs upstream):
  - 线性backcast → 2层MLP + tanh门控
  - backcast = tanh(gate) * MLP(hidden), 门控学习哪些信息需要回溯
"""
import os
import sys
import torch
import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="DifBlock"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class TanhGatedMLP(nn.Module):
    """2层MLP + tanh门控: output = tanh(gate_mlp(x)) * value_mlp(x)"""

    def __init__(self, hidden_dim):
        super().__init__()
        # value path
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.value_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # gate path
        self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # value branch
        v = self.act(self.value_fc1(x))
        v = self.value_fc2(v)
        # gate branch
        g = self.act(self.gate_fc1(x))
        g = torch.tanh(self.gate_fc2(g))
        return g * v


class DifBlock(nn.Module):
    """CardGame Diffusion Block with tanh-gated MLP backcast"""

    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
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

        # CardGame改写: 2层MLP + tanh门控替代线性backcast
        self.backcast_branch = TanhGatedMLP(hidden_dim)

        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data,
                dynamic_graph, static_graph):
        _dbg("input.history", history_data)
        _dbg("input.gated", gated_history_data)

        # diffusion model
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)
        _dbg("hidden_states_dif", hidden_states_dif)

        # forecast branch
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        # CardGame: tanh gated MLP backcast
        backcast_seq = self.backcast_branch(hidden_states_dif)
        _dbg("backcast_gated", backcast_seq)

        # residual decomposition
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)
        _dbg("backcast_residual", backcast_seq_res)

        return backcast_seq_res, forecast_hidden
