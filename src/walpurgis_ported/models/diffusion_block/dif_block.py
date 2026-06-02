"""
Walpurgis v4 Diffusion Block — Pre-Conv RMSNorm + Gradient Sentinel
=========================================================================
Delta vs v3:
  - LayerNorm → *RMSNorm + scaled dropout*.  RMSNorm (Zhang &
    Sennrich 2019) omits the mean-centering step, reducing compute
    by ~15% while providing comparable stabilisation for the
    gated input distribution before conv.
  - Added gradient sentinel: registers a backward hook that logs
    gradient statistics every 500 steps for dead-neuron detection.
  - Norm ratio tracking now uses EMA for smoother trend.

Breakpoint helpers:
    self._diag_last          # dict with last forward stats
    self._grad_sentinel_log  # list of recent gradient snapshots
"""
import time
import torch
import torch.nn as nn
from collections import deque
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
        # Pre-conv LayerNorm + scaled dropout
        # RMSNorm: simpler than LayerNorm, ~15% faster (no mean subtraction)
        self._pre_rms_weight = nn.Parameter(torch.ones(hidden_dim))
        self._pre_rms_eps = 1e-6
        self._pre_drop = nn.Dropout(0.06)
        self._debug = True
        self._ema_ratio = 1.0
        self._ema_coeff = 0.05
        self._diag_last = {}
        self._grad_sentinel_log = deque(maxlen=50)

        # Register gradient sentinel hook
        self.conv.spatial_proj.weight.register_hook(self._grad_sentinel_hook)

    def _grad_sentinel_hook(self, grad):
        """Backward hook: log gradient health for dead-neuron detection."""
        if DifBlock._n % 500 == 0:
            with torch.no_grad():
                gn = grad.norm().item()
                zero_frac = (grad.abs() < 1e-8).float().mean().item()
                self._grad_sentinel_log.append({
                    "step": DifBlock._n,
                    "grad_norm": round(gn, 6),
                    "dead_frac": round(zero_frac, 4),
                })
                if zero_frac > 0.3:
                    print(
                        f"      [DifBlock SENTINEL] ⚠ {zero_frac*100:.1f}% dead neurons "
                        f"in spatial_proj at step {DifBlock._n}"
                    )

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        DifBlock._n += 1
        verbose = self._debug and DifBlock._n % 500 == 1
        if verbose:
            print(
                f"      [DifBlock #{DifBlock._n}] in={list(history_data.shape)} "
                f"gated={list(gated_history_data.shape)}"
            )

        # LayerNorm + dropout instead of raw dropout
        # RMSNorm: x / sqrt(mean(x²) + eps) * weight
        rms = gated_history_data.float().pow(2).mean(-1, keepdim=True).add(self._pre_rms_eps).sqrt()
        gated_stable = (gated_history_data / rms) * self._pre_rms_weight
        gated_drop = self._pre_drop(gated_stable)

        conv_out = self.conv(gated_drop, dynamic_graph, static_graph)
        forecast_h = self.forecast_branch(conv_out)
        backcast_r = self.residual_decomp(
            history_data[:, -conv_out.shape[1]:], conv_out,
        )

        if self._debug:
            with torch.no_grad():
                ratio = conv_out.norm().item() / (gated_history_data.norm().item() + 1e-8)
                self._ema_ratio = self._ema_coeff * ratio + (1 - self._ema_coeff) * self._ema_ratio
                self._diag_last = {
                    "step": DifBlock._n,
                    "norm_ratio": round(ratio, 4),
                    "ema_ratio": round(self._ema_ratio, 4),
                    "conv_out_norm": round(conv_out.norm().item(), 4),
                }
        if verbose:
            d = self._diag_last
            print(
                f"      [DifBlock] out: back={list(backcast_r.shape)} "
                f"fc={list(forecast_h.shape)} ratio={d['norm_ratio']:.4f} "
                f"ema_ratio={d['ema_ratio']:.4f}"
            )
        return backcast_r, forecast_h
