"""
扩散块子包 — walpurgis 实现
====================
upstream 直接 from .dif_block import DifBlock, 无守卫.
改动:
  1) 延迟导入守卫: 捕获 import 失败并给出明确提示
  2) 模块级 inspect 函数: dump_dif_block_state() 打印所有子模块参数形状
"""
from walpurgis_walking import _dbg

_TAG = "dif_init"

try:
    from .dif_block import DifBlock
except ImportError as _e:
    raise ImportError(
        f"[walpurgis] DifBlock 导入失败 — 检查 dif_block.py 依赖: {_e}"
    ) from _e

_dbg(_TAG, "diffusion_block package loaded")


def dump_dif_block_state(block: DifBlock, prefix: str = ""):
    """断点调试辅助: 打印 DifBlock 所有子模块参数的形状和统计量."""
    import torch as _th
    print(f"{'=' * 60}")
    print(f"[walpurgis DifBlock Inspector] {prefix}")
    for name, param in block.named_parameters():
        grad_info = "no_grad"
        if param.grad is not None:
            grad_info = (f"grad_norm={param.grad.norm().item():.4g} "
                         f"grad_max={param.grad.abs().max().item():.4g}")
        print(f"  {name:40s} | shape={str(tuple(param.shape)):20s} "
              f"| mean={param.data.mean().item():.4g} "
              f"| std={param.data.std().item():.4g} "
              f"| {grad_info}")
    print(f"{'=' * 60}")


__all__ = ["DifBlock", "dump_dif_block_state"]
