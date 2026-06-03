"""
Walpurgis-Ported v6 — Decoupled Spatio-Temporal Graph Neural Network
=====================================================================
Re-engineered variant with enhanced diagnostic instrumentation.
Every submodule exposes ``_debug_snapshot()`` helpers so that intermediate
tensor states can be dumped at any granularity during training / inference.

Changelog vs upstream d2stgnn
-----------------------------
* Gating activation swapped to GELU in estimation gate (smoother gradient).
* Forecast branch uses adaptive-gap stride instead of fixed gap.
* Laplacian computation adds eps-stabilised inversion path.
* Full lifecycle ``print`` instrumentation (shape, dtype, min/max/mean)
  controlled via ``WALPURGIS_DEBUG`` environment variable.
"""

__version__ = "6.0.0-walpurgis"

import os as _os

# ── runtime self-check ──────────────────────────────────────────────
WALPURGIS_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0") == "1"

def _dbg(tag: str, obj=None, **kw):
    """Conditional debug printer.  Enable with  WALPURGIS_DEBUG=1 ."""
    if not WALPURGIS_DEBUG:
        return
    parts = [f"[WPG-DBG][{tag}]"]
    if obj is not None:
        import torch
        if isinstance(obj, torch.Tensor):
            parts.append(
                f"shape={list(obj.shape)} dtype={obj.dtype} "
                f"min={obj.min().item():.6f} max={obj.max().item():.6f} "
                f"mean={obj.float().mean().item():.6f}"
            )
        else:
            parts.append(repr(obj)[:200])
    for k, v in kw.items():
        parts.append(f"{k}={v}")
    print(" | ".join(parts), flush=True)
