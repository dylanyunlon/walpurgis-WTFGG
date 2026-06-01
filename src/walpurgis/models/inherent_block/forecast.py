"""
Walpurgis v2 Inherent Forecast — Selective Checkpoint with Norm Monitor
==========================================================================
Delta vs prior:
  - Gradient checkpointing: always-on → *selective* — only checkpoints
    when the hidden dimension exceeds a threshold (256), avoiding the
    recomputation overhead for small models.
  - Added residual connection from input to output with a learnable
    mixing coefficient α, providing a gradient highway through the
    forecast head.
  - Detailed norm tracking at each stage for gradient flow diagnosis.

Breakpoint helpers:
    self._diag_last       # dict with last forward stats
    self._alpha_value()   # current residual mixing coefficient
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Forecast(nn.Module):
    _n = 0

    def __init__(self, hidden_dim, forecast_dim, output_seq_len=12, gap=1):
        super().__init__()
        self.fc_in = nn.Linear(hidden_dim, forecast_dim)
        self.fc_out = nn.Linear(forecast_dim, output_seq_len // gap)
        self.drop = 0.1
        # Learnable residual mixing: output = α·skip + (1-α)·transformed
        self._res_alpha = nn.Parameter(torch.tensor(0.1))
        self._ckpt_threshold = 256
        self._debug = True
        self._diag_last = {}

    def _alpha_value(self):
        """Current residual mixing coefficient — call from pdb."""
        return torch.sigmoid(self._res_alpha).item()

    def _inner(self, h):
        h = F.relu(self.fc_in(h))
        if self.training:
            h = F.dropout(h, p=self.drop)
        return h

    def forward(self, hidden):
        Forecast._n += 1
        h = hidden.mean(dim=1)
        skip = h  # save for residual

        # Selective checkpointing: only for large hidden dims
        use_ckpt = (self.training and h.requires_grad
                    and h.shape[-1] >= self._ckpt_threshold)
        if use_ckpt:
            h = checkpoint(self._inner, h, use_reentrant=False)
        else:
            h = self._inner(h)

        # Learnable residual mixing
        alpha = torch.sigmoid(self._res_alpha)
        if skip.shape[-1] != h.shape[-1]:
            # Dimension mismatch: project skip
            skip_proj = F.linear(skip, self.fc_in.weight[:, :skip.shape[-1]])
            h = alpha * skip_proj + (1.0 - alpha) * h
        else:
            h = alpha * skip + (1.0 - alpha) * h

        if self._debug:
            with torch.no_grad():
                self._diag_last = {
                    "step": Forecast._n,
                    "alpha": round(alpha.item(), 4),
                    "h_norm": round(h.norm().item(), 4),
                    "checkpointed": use_ckpt,
                }
            if Forecast._n % 1000 == 1:
                d = self._diag_last
                ckpt_tag = "✓ckpt" if d["checkpointed"] else "no_ckpt"
                print(
                    f"        [InhForecast #{Forecast._n}] "
                    f"α_res={d['alpha']:.4f} h_‖={d['h_norm']:.4f} [{ckpt_tag}]"
                )
        return h
