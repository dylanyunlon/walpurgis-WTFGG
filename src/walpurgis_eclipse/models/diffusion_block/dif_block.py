"""Eclipse DifBlock: 2-layer MLP + sigmoid gated backcast."""
import torch, torch.nn as nn, sys, os
from .forecast import Forecast; from .dif_model import STLocalizedConv; from ..decouple.residual_decomp import ResidualDecomp
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        self.forecast_dropout = nn.Dropout(0.1)
        self.backcast_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.backcast_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hsd = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        fk = self.forecast_dropout(self.forecast_branch(gated_history_data, hsd, self.localized_st_conv, dynamic_graph, static_graph))
        gate = self.backcast_gate(hsd); bc = gate * self.backcast_mlp(hsd)
        hd = history_data[:, -bc.shape[1]:, :, :]
        res = self.residual_decompose(hd, bc)
        if _ECL_DBG: print(f"[ECL:difblk] bc_e={bc.norm().item():.4f} fk_e={fk.norm().item():.4f} gate={gate.mean().item():.4f}", file=sys.stderr)
        return res, fk
