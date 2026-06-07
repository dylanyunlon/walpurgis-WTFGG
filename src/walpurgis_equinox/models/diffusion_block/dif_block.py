import torch
import torch.nn as nn
import torch.nn.utils as nnutils
import sys, os

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:difblk:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DifBlock(nn.Module):
    """upstream: 单层Linear backcast
    equinox: 2层WeightNorm-MLP+Mish门控backcast, Dropout在forecast前"""
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(hidden_dim,
                                        forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # upstream: 单层 Linear
        # equinox: 2层WeightNorm-MLP + Mish门控
        self.bc_fc1 = nnutils.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.bc_fc2 = nnutils.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.bc_gate = nn.Linear(hidden_dim, hidden_dim)
        self.bc_act = nn.Mish()
        # equinox: forecast前Dropout
        self.forecast_dropout = nn.Dropout(0.1)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        # equinox: forecast前加dropout
        hidden_dropped = self.forecast_dropout(hidden_dif)
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_dropped, self.localized_st_conv,
            dynamic_graph, static_graph)
        # equinox: 2层WeightNorm-MLP + 门控backcast
        bc = self.bc_act(self.bc_fc1(hidden_dif))
        gate = torch.sigmoid(self.bc_gate(hidden_dif))
        backcast_seq = gate * self.bc_fc2(bc)
        _edbg("bc_gate_mean", gate.mean())
        _edbg("bc_vs_fk", torch.norm(backcast_seq) / (torch.norm(forecast_hidden) + 1e-8))

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_res = self.residual_decompose(history_data, backcast_seq)
        return backcast_res, forecast_hidden
