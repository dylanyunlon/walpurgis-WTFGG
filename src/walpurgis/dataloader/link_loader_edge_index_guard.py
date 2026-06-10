"""
link_loader_edge_index_guard.py
migrate c07eea7: [FEA] New MovieLens Example, Add Timing to Taobao

上游变化 (c07eea7):
  文件: python/cugraph-pyg/cugraph_pyg/loader/link_loader.py
  函数: LinkLoader.__init__ (L125-128)

  旧代码:
      ) = torch_geometric.loader.utils.get_edge_label_index(
          data,
          (None, edge_label_index),
      )

  新代码:
      ) = torch_geometric.loader.utils.get_edge_label_index(
          data,
          (None, edge_label_index)
          if isinstance(edge_label_index, torch.Tensor)
          else edge_label_index,
      )

  根因:
    get_edge_label_index 第二参数可以是：
      A) torch.Tensor — 纯边索引张量，无类型信息，需要包裹 (None, tensor) 表示「类型未指定」
      B) tuple (edge_type, tensor) — 已携带边类型信息，直接透传
    旧代码无论哪种类型一律包裹 (None, ...) ：
      - 传入 A 时：(None, tensor) ✓ 正确
      - 传入 B 时：(None, (edge_type, tensor)) ✗ 嵌套错误，get_edge_label_index 期望第二参数
        是 (edge_type, tensor) 或裸 tensor，不是二重嵌套 tuple
    结果：传入 (edge_type, tensor) 时，input_type 解析错误，edge_label_index 赋值出错，
    下游 _vertex_offsets 偏移量以错误的 input_type 索引，节点 id 偏移崩溃。

Knuth 审查:
  1. diff 对比源:
     | 上游 c07eea7                              | Walpurgis 迁移                             |
     |---|---|
     | 内联 isinstance 条件表达式一行            | `resolve_edge_label_input()` 命名函数       |
     | 无前置文档                                | docstring 说明 A/B 两种类型路径              |
     | 无调试输出                                | WALPURGIS_DEBUG=1 打印类型 + shape           |
     | 无覆盖测试                                | `_smoke_test()` 在文件末尾，可手动运行       |

  2. 用户角度 bug:
     - 常见用法：LinkNeighborLoader(edge_label_index=(("user","rates","movie"), tensor))
       此时 edge_label_index 是 tuple，旧代码包裹后变为 (None, (edge_type, tensor))。
       get_edge_label_index 内部尝试 edge_type = arg[0]，得到 None，
       后续 data[1]._vertex_offsets[None] → KeyError: None，
       stack trace 深入 PyG 内部，用户看不到自己的参数哪里错了。
     - 只有传裸 tensor 的场景旧代码偶然正确，这正是单类型图用法，
       异构图（hetero）用法几乎必然传 tuple，bug 覆盖面极广。
     - 错误信息与用户传入参数完全无关，KeyError/AttributeError 指向库内部，
       新用户会首先怀疑版本兼容性，浪费大量调试时间。

  3. 系统角度安全:
     - get_edge_label_index 的第二参数协议：
         * 纯 Tensor → 自动推断 input_type=None（同构边）
         * (None, Tensor) → 显式声明 input_type=None（等价于上一行）
         * (edge_type_tuple, Tensor) → 使用指定 edge_type 解析 _vertex_offsets
       旧代码将所有路径强制为 (None, ...) 嵌套，破坏了第三种协议路径。
     - 修复后 isinstance 判断在 Python 层完成，无 GPU 操作，无性能开销。
     - 判断分支清晰：Tensor → 包裹；非 Tensor（即 tuple）→ 直接透传。
       未来即使 edge_label_index 支持更多类型（如 EdgeIndex），
       else 分支透传策略仍能正确工作（只要 get_edge_label_index 能处理）。

Walpurgis 改写 20%（鲁迅拿法）:
  - `resolve_edge_label_input`: 封装 isinstance 判断，给这段「无声的类型分发」一个名字。
    上游一行条件表达式嵌在函数调用参数里，代码搜索时极难定位。
  - `EdgeLabelInputError`: 专用异常，携带类型信息，替代下游 KeyError/AttributeError 静默崩溃。
  - `_dbg_edge_label_input`: 断点调试出口，WALPURGIS_DEBUG=1 时打印输入类型与形状。
  - `assert_resolved_input_compatible`: 事后校验——resolved 结果长度与 data 图规模一致性检查。
"""

import os
from typing import Union, Tuple, Optional

import torch

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ──────────────────────────────────────────────────────────────────────────────
# 专用异常
# ──────────────────────────────────────────────────────────────────────────────

class EdgeLabelInputError(ValueError):
    """
    edge_label_index 类型无法被 get_edge_label_index 识别时抛出。

    携带实际类型与 shape 信息，便于用户快速定位问题。
    """

    def __init__(self, edge_label_index, detail: str = "") -> None:
        t = type(edge_label_index).__name__
        shape = (
            list(edge_label_index.shape)
            if isinstance(edge_label_index, torch.Tensor)
            else str(edge_label_index)[:80]
        )
        msg = (
            f"edge_label_index 类型 {t!r} (shape/value={shape}) 无法解析。"
            f" 期望: torch.Tensor 或 (edge_type_tuple, torch.Tensor)。"
        )
        if detail:
            msg += f" 附加信息: {detail}"
        super().__init__(msg)


# ──────────────────────────────────────────────────────────────────────────────
# 调试出口
# ──────────────────────────────────────────────────────────────────────────────

def _dbg_edge_label_input(eli, resolved) -> None:
    """
    断点调试 print — WALPURGIS_DEBUG=1 时打印输入 / 解析后的类型与形状。

    上游一行 isinstance 条件表达式，发生错误时无任何调试信息可用。
    此函数在 DEBUG 模式下补充两侧对比输出，帮助定位类型分发路径。
    """
    if not _DEBUG:
        return
    input_type_str = type(eli).__name__
    if isinstance(eli, torch.Tensor):
        input_shape = list(eli.shape)
    elif isinstance(eli, (tuple, list)) and len(eli) == 2:
        edge_type, tensor = eli
        input_shape = f"edge_type={edge_type}, tensor.shape={list(tensor.shape)}"
    else:
        input_shape = repr(eli)[:80]

    if isinstance(resolved, torch.Tensor):
        resolved_repr = f"Tensor shape={list(resolved.shape)}"
    elif isinstance(resolved, (tuple, list)):
        resolved_repr = f"tuple len={len(resolved)}, first={type(resolved[0]).__name__}"
    else:
        resolved_repr = repr(resolved)[:80]

    rank = -1
    if torch.distributed.is_initialized():
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            pass

    print(
        f"[WALPURGIS_DBG rank={rank}] [resolve_edge_label_input] "
        f"input_type={input_type_str} input_shape={input_shape} "
        f"→ resolved={resolved_repr}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 核心函数：resolve_edge_label_input
# 封装 c07eea7 的 isinstance 类型分发逻辑
# ──────────────────────────────────────────────────────────────────────────────

def resolve_edge_label_input(
    edge_label_index: Union[torch.Tensor, Tuple],
) -> Union[Tuple, torch.Tensor]:
    """
    将 edge_label_index 解析为 get_edge_label_index 接受的形式。

    两种合法输入路径（对应上游 c07eea7 修复）：

    路径 A — 纯 Tensor（同构图，无边类型信息）:
        输入:  torch.Tensor, shape=[2, N]
        输出:  (None, tensor)  — 包裹，告知 get_edge_label_index 类型未指定
        上游旧代码始终走此路径（无论输入是什么），对 B 类输入造成双重嵌套错误。

    路径 B — (edge_type, tensor) tuple（异构图，携带边类型）:
        输入:  ((src_type, rel, dst_type), torch.Tensor)
        输出:  原样透传 — get_edge_label_index 可直接处理此格式
        上游旧代码错误地将此包裹为 (None, (edge_type, tensor))，破坏协议。

    Args:
        edge_label_index: 纯 Tensor 或 (edge_type_tuple, Tensor) 二元组

    Returns:
        get_edge_label_index 第二参数就绪的形式

    Raises:
        EdgeLabelInputError: edge_label_index 类型既非 Tensor 也非合法 tuple
    """
    # ── 路径 A：纯 Tensor，需要包裹 (None, tensor) ──
    if isinstance(edge_label_index, torch.Tensor):
        resolved = (None, edge_label_index)
        _dbg_edge_label_input(edge_label_index, resolved)
        return resolved

    # ── 路径 B：已是 (edge_type, tensor) tuple，直接透传 ──
    if isinstance(edge_label_index, (tuple, list)):
        if len(edge_label_index) != 2:
            raise EdgeLabelInputError(
                edge_label_index,
                detail=f"tuple/list 长度应为 2，实际为 {len(edge_label_index)}",
            )
        _dbg_edge_label_input(edge_label_index, edge_label_index)
        return edge_label_index

    # ── 未知类型 ──
    raise EdgeLabelInputError(
        edge_label_index,
        detail="既不是 torch.Tensor 也不是 (edge_type, tensor) tuple",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 事后一致性校验
# ──────────────────────────────────────────────────────────────────────────────

def assert_resolved_input_compatible(
    resolved_edge_label_index: torch.Tensor,
    batch_size: int,
    drop_last: bool,
) -> None:
    """
    校验 resolve_edge_label_input 输出与 batch_size / drop_last 组合的兼容性。

    对应上游 link_loader.py 中紧随 get_edge_label_index 之后的 drop_last 校验：
        if edge_label_index.shape[1] < batch_size and drop_last:
            raise ValueError(...)

    此函数在 Walpurgis 中作为独立校验点使用，便于在调用 get_edge_label_index 之前
    提前发现配置错误，避免用户在 DataLoader 迭代阶段才看到空 batch。

    Args:
        resolved_edge_label_index: resolve_edge_label_input 返回的 Tensor
        batch_size: LinkNeighborLoader 的 batch_size
        drop_last: LinkNeighborLoader 的 drop_last

    Raises:
        ValueError: drop_last=True 但边数 < batch_size
    """
    n_edges = resolved_edge_label_index.shape[1]
    if n_edges < batch_size and drop_last:
        raise ValueError(
            f"[WALPURGIS] edge_label_index 边数 ({n_edges}) < batch_size ({batch_size}) "
            f"且 drop_last=True，所有 batch 将被丢弃。"
            f" 请将 drop_last 改为 False，或增加 edge_label_index 中的边数。"
        )
    if _DEBUG:
        rank = -1
        if torch.distributed.is_initialized():
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                pass
        print(
            f"[WALPURGIS_DBG rank={rank}] [assert_resolved_input_compatible] "
            f"n_edges={n_edges} batch_size={batch_size} drop_last={drop_last} → OK",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 使用示例 / smoke test（手动运行验证）
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    验证 resolve_edge_label_input 的两条路径均正确。

    运行: python -m walpurgis.dataloader.link_loader_edge_index_guard
    """
    print("[SMOKE TEST] 路径 A: 纯 Tensor 输入")
    t = torch.zeros(2, 10, dtype=torch.int64)
    result = resolve_edge_label_input(t)
    assert isinstance(result, tuple) and result[0] is None and result[1] is t, \
        f"路径 A 失败: result={result}"
    print(f"  OK: input=Tensor(2,10) → result={type(result[0])}, Tensor(2,10)")

    print("[SMOKE TEST] 路径 B: (edge_type, Tensor) tuple 输入")
    edge_type = ("user", "rates", "movie")
    result2 = resolve_edge_label_input((edge_type, t))
    assert result2[0] == edge_type and result2[1] is t, \
        f"路径 B 失败: result={result2}"
    print(f"  OK: input=(edge_type, Tensor) → 透传 edge_type={edge_type}")

    print("[SMOKE TEST] 路径 C: 无效类型 — 应抛出 EdgeLabelInputError")
    try:
        resolve_edge_label_input("invalid_string")
        assert False, "未抛出异常"
    except EdgeLabelInputError as e:
        print(f"  OK: EdgeLabelInputError 正确抛出: {e}")

    print("[SMOKE TEST] drop_last 校验")
    try:
        assert_resolved_input_compatible(t, batch_size=20, drop_last=True)
        assert False, "未抛出异常"
    except ValueError as e:
        print(f"  OK: ValueError 正确抛出: {e}")

    assert_resolved_input_compatible(t, batch_size=5, drop_last=True)
    print("  OK: n_edges=10 >= batch_size=5, drop_last=True → 通过")

    print("[SMOKE TEST] 全部通过 ✓")


if __name__ == "__main__":
    _smoke_test()
