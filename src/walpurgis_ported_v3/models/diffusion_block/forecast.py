"""
Auto-regressive forecast head for the diffusion branch.
"""
import sys
import torch
import torch.nn as nn

_DBG = ("--debug-diffc" in sys.argv)


class Forecast(nn.Module):
    """Recursively predict future hidden states via the localized ST conv."""

    def __init__(self, hidden_dim, forecast_hidden_dim=None, **kw):
        super().__init__()
        self.k_t = kw['k_t']
        self.horizon = kw['seq_length']
        self.gap = kw['gap']
        self.proj = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(self, gated_hist, diff_hidden, st_conv, dyn_g, sta_g):
        preds = [diff_hidden[:, -1, :, :].unsqueeze(1)]
        n_steps = int(self.horizon / self.gap) - 1

        for step_i in range(n_steps):
            tail = preds[-self.k_t:]
            if len(tail) < self.k_t:
                # borrow from gated_hist to fill the initial window
                need = self.k_t - len(tail)
                prefix = gated_hist[:, -need:, :, :]
                tail = [prefix] + tail
            combined = torch.cat(tail, dim=1)
            next_h = st_conv(combined, dyn_g, sta_g)
            preds.append(next_h)

            if _DBG and step_i < 2:
                print(f"[DBG:diffc] step {step_i}  "
                      f"combined={tuple(combined.shape)}  "
                      f"next_h_mean={next_h.mean().item():.4f}")

        out = torch.cat(preds, dim=1)
        out = self.proj(out)
        return out
