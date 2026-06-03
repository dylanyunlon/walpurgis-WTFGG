import torch.nn as nn
import sys

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_DBG = ("--dbg" in sys.argv)


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256,
                 use_pre=None, dy_graph=None, sta_graph=None,
                 **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(
            hidden_dim, pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph,
            **model_args)

        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim=forecast_hidden_dim,
            **model_args)

        # 算法改动: backcast 分支用 GLU (Gated Linear Unit)
        # 原版: 单个 Linear(hidden, hidden)
        # 改为: Linear(hidden, 2*hidden) 然后 split -> sigmoid(a) * b
        # GLU 自带 gating, 选择性地保留信息, 比纯线性更有表达力
        self.backcast_gate = nn.Linear(hidden_dim, hidden_dim * 2)

        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def _glu_backcast(self, x):
        h = self.backcast_gate(x)
        a, b = h.chunk(2, dim=-1)
        return a.sigmoid() * b

    def forward(self, history_data, gated_history_data,
                dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(
            gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states_dif,
            self.localized_st_conv, dynamic_graph, static_graph)

        # GLU backcast
        backcast_seq = self._glu_backcast(hidden_states_dif)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(
            history_data, backcast_seq)

        if _DBG:
            import torch
            with torch.no_grad():
                print(f"[DBG][DifBlock] forecast shape={list(forecast_hidden.shape)}  "
                      f"backcast_res shape={list(backcast_seq_res.shape)}  "
                      f"backcast_mean={backcast_seq.mean().item():.5f}", flush=True)
        return backcast_seq_res, forecast_hidden
