"""
动态图构建子包 — walpurgis 实现
========================
upstream 原样导出; 加入导入守卫 + 图结构 inspector.
"""
from walpurgis_walking import _dbg

_TAG = "dygraph_init"

try:
    from .dy_graph_conv import DynamicGraphConstructor
except ImportError as _e:
    raise ImportError(
        f"[walpurgis] DynamicGraphConstructor 导入失败: {_e}"
    ) from _e

_dbg(_TAG, "dynamic_graph_conv package loaded")


def dump_graph_constructor(ctor: DynamicGraphConstructor, prefix: str = ""):
    """断点调试: 打印动态图构建器内部状态 — 时间权重、cos投影矩阵等."""
    import torch as _th
    print(f"{'=' * 60}")
    print(f"[walpurgis DynGraph Inspector] {prefix}")
    # 时间权重 softmax
    t_w = _th.nn.functional.softmax(ctor.temporal_logits, dim=0)
    print(f"  temporal_logits (raw):  {ctor.temporal_logits.data.tolist()}")
    print(f"  temporal_weights (sm):  {[f'{w:.4f}' for w in t_w.tolist()]}")
    print(f"  cos_proj weight norm:   {ctor.cos_proj.weight.data.norm().item():.4g}")
    for name, p in ctor.named_parameters():
        if p.grad is not None:
            print(f"  {name:30s} grad_norm={p.grad.norm().item():.4g}")
    print(f"{'=' * 60}")


__all__ = ["DynamicGraphConstructor", "dump_graph_constructor"]
