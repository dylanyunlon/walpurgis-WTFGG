"""
walpurgis_ported_v9  —  D2STGNN algorithmic port (v9)
=====================================================
Port lineage : upstream/d2stgnn → walpurgis_ported_v9
Algorithmic delta ≈ 20 %  (see per-file docstrings for specifics)

Global debug toggle
-------------------
Set env  WALPURGIS_V9_DEBUG=1  to activate *all* _dbg() calls.
Each module also honours a fine-grained --debug-<tag> CLI flag.
"""
import os as _os

_GLOBAL_DEBUG = _os.environ.get("WALPURGIS_V9_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, *, force: bool = False):
    """Unified debug printer.  Active when WALPURGIS_V9_DEBUG=1 or *force*."""
    if _GLOBAL_DEBUG or force:
        print(f"[v9-dbg][{tag}] {msg}")
