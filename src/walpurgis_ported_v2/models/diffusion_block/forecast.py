"""
Auto-regressive forecast branch for the Diffusion Block.
Generates future hidden states by iteratively applying the
ST-localized convolution.
"""

import torch
import torch.nn as nn
import sys

_DBG_DFCAST = ("--debug-dfcast" in sys.argv) or False


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.horizon = model_args['seq_length']
        self.gap = model_args['gap']
        self.proj = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(self, gated_history, dif_hidden, st_conv, dynamic_graph, static_graph):
        """
        Auto-regressively roll out future steps using the ST conv.

        Parameters
        ----------
        gated_history : [B, L, N, D]   — gated input sequence
        dif_hidden    : [B, L', N, D]  — output of ST conv on current input
        st_conv       : STLocalizedConv module (reused for AR steps)
        dynamic_graph, static_graph : graph support sets

        Returns
        -------
        [B, horizon//gap, N, forecast_dim]
        """
        n_ar_steps = int(self.horizon / self.gap) - 1
        predictions = [dif_hidden[:, -1, :, :].unsqueeze(1)]

        for step in range(n_ar_steps):
            # gather recent k_t predictions (pad with history if not enough)
            recent = predictions[-self.k_t:]
            if len(recent) < self.k_t:
                n_pad = self.k_t - len(recent)
                pad = gated_history[:, -n_pad:, :, :]
                recent = [pad] + recent
            ar_input = torch.cat(recent, dim=1)
            next_h = st_conv(ar_input, dynamic_graph, static_graph)
            predictions.append(next_h)

            if _DBG_DFCAST:
                print(f"[DBG:dfcast] AR step {step+1}/{n_ar_steps}  "
                      f"input_len={ar_input.shape[1]}  "
                      f"next_h_norm={next_h.norm().item():.4f}")

        forecast = torch.cat(predictions, dim=1)
        forecast = self.proj(forecast)

        if _DBG_DFCAST:
            print(f"[DBG:dfcast] final  shape={tuple(forecast.shape)}  "
                  f"norm={forecast.norm().item():.4f}")
        return forecast
