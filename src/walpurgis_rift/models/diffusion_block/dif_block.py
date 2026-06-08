"""Rift DifBlock: FFT-enhanced backcast mechanism.
Unlike upstream (single FC backcast), Rift augments the backcast branch with FFT:
the backcast is computed from both time-domain and frequency-domain representations,
then fused via a learned gate."""
import torch, torch.nn as nn, sys, os
from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp

_RF_DBG = os.environ.get('RIFT_DEBUG', '0') == '1'


class FFTBackcast(nn.Module):
    """Rift特有: FFT增强的backcast分支. 时域FC + 频域FC, 门控融合"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_fc = nn.Linear(hidden_dim, hidden_dim)
        self.freq_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        time_out = self.time_fc(x)
        freq_out_spec = torch.fft.rfft(self.freq_fc(x), dim=1)
        freq_out = torch.fft.irfft(freq_out_spec, n=x.shape[1], dim=1)
        gate = torch.sigmoid(self.fusion_gate)
        out = (1 - gate) * time_out + gate * freq_out
        if _RF_DBG:
            print(f"[RF-DBG:fft_backcast] gate={gate.item():.4f} "
                  f"time_norm={time_out.norm().item():.4f} "
                  f"freq_norm={freq_out.norm().item():.4f}",
                  file=sys.stderr, flush=True)
        return out


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        self.backcast_branch = FFTBackcast(hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        backcast_seq = self.backcast_branch(hidden_states_dif)
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)
        if _RF_DBG:
            print(f"[RF-DBG:dif_block] dif_out={hidden_states_dif.shape} "
                  f"fk={forecast_hidden.shape} bc_res={backcast_seq_res.shape}",
                  file=sys.stderr, flush=True)
        return backcast_seq_res, forecast_hidden
