import torch
import torch.nn as nn
import sys, os

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:difblk:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DifBlock(nn.Module):
    """upstream: 单层Linear backcast
    aurora: 2层MLP+GroupNorm+sigmoid门控backcast, Dropout在forecast前"""
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
        # aurora: 2层MLP + GroupNorm + sigmoid门控
        self.bc_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bc_gn = nn.GroupNorm(min(4, hidden_dim), hidden_dim)
        self.bc_gate = nn.Linear(hidden_dim, hidden_dim)
        self.bc_act = nn.GELU()
        # aurora: forecast前Dropout
        self.forecast_dropout = nn.Dropout(0.1)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        # aurora: forecast前加dropout
        hidden_dropped = self.forecast_dropout(hidden_dif)
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_dropped, self.localized_st_conv,
            dynamic_graph, static_graph)
        # aurora: 2层MLP + 门控backcast
        bc = self.bc_act(self.bc_fc1(hidden_dif))
        bc = bc.permute(0, 3, 1, 2)
        bc = self.bc_gn(bc).permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.bc_gate(hidden_dif))
        backcast_seq = gate * self.bc_fc2(bc)
        _adbg("bc_gate_mean", gate.mean())
        _adbg("bc_vs_fk", torch.norm(backcast_seq) / (torch.norm(forecast_hidden) + 1e-8))

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_res = self.residual_decompose(history_data, backcast_seq)
        return backcast_res, forecast_hidden
