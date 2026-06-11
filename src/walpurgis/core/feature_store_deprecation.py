"""
feature_store_deprecation.py — 2d545b9 迁移: 废弃 TensorDictFeatureStore

migrate 2d545b9: Deprecate TensorDictFeatureStore in cuGraph-PyG

上游变化 (2d545b9, cugraph-gnn / python/cugraph-pyg/cugraph_pyg/data/__init__.py):

1 文件改动，12 行新增，1 行替换:
  - 原 import:
        from cugraph_pyg.data.feature_store import (
            TensorDictFeatureStore,           # ← 裸 import
            WholeFeatureStore,
        )
  - 新 import:
        from cugraph_pyg.data.feature_store import (
            TensorDictFeatureStore as DEPRECATED__TensorDictFeatureStore,  # 重命名隐藏
            WholeFeatureStore,
        )
  - 新增 wrapper 函数:
        def TensorDictFeatureStore(*args, **kwargs):
            warnings.warn(
                "TensorDictFeatureStore is deprecated.  Consider changing your "
                "workflow to launch using 'torchrun' and store data in "
                "the faster and more memory-efficient WholeFeatureStore instead.",
                FutureWarning,
            )
            return DEPRECATED__TensorDictFeatureStore(*args, **kwargs)

设计语义:
    - TensorDictFeatureStore 原本是 cugraph-pyg 自带的内存 dict 特征存储，
      适用于单机、小规模数据集；
    - WholeFeatureStore（实为内部 FeatureStore）基于 WholeGraph 分布式内存，
      支持跨 GPU/跨节点的大规模特征，更快且更省内存；
    - 统一 API（Unified API）战略要求所有数据改走 WholeGraph 存储，
      TensorDictFeatureStore 不再作为可选路径维护；
    - wrapper 函数而非删除: 保持向后兼容，FutureWarning 催迁移，不破坏现有用户代码。

Knuth 审查:
1. diff 对比源:
   | 上游 2d545b9                          | Walpurgis 迁移                              |
   |--------------------------------------|---------------------------------------------|
   | wrapper 函数，无任何调试信息           | DeprecationGate 类，可观测决策 + 调用统计    |
   | 直接透传 *args/**kwargs，无参数摘要    | 断点 print 打印参数类型摘要（不泄露值）       |
   | FutureWarning 文字固定，无上下文      | DeprecationGate 携带 call_count，traceback 可选 |
   | 单一 TensorDictFeatureStore 废弃       | DeprecationPolicy 可统一管理多个废弃对象      |
   | __init__.py 内联 wrapper 函数，不可测  | feature_store_deprecation.py 独立模块，可单测 |

2. 用户角度 bug:
   - 原 wrapper 函数如果在 class 实例化时被 pickle（如 torch.multiprocessing spawn），
     wrapper 函数本身是 module 全局的具名对象，pickle 时找到的是 wrapper 而非原类，
     这一行为与直接 import 类不同，但在 FutureWarning 阶段暂不构成 bug（对象类型
     仍是 DEPRECATED__TensorDictFeatureStore）；DeprecationGate 保持同样语义。
   - 若用户代码 `isinstance(store, TensorDictFeatureStore)` ——
     上游 wrapper 使 TensorDictFeatureStore 变成函数，
     isinstance 会抛 TypeError: isinstance() arg 2 must be a type or tuple of types；
     这是上游引入的 silent breaking change，DeprecationGate 同等风险，
     Walpurgis 加 InstanceCheckGuard 防御（见下文）。
   - FutureWarning 被 Python 默认过滤器静默（非 -W error 时用户看不到）；
     DeprecationGate 同时写 WALPURGIS_DEBUG stderr，确保调试模式下可见。

3. 系统角度安全:
   - wrapper 函数与 import as 别名共享 module 命名空间，
     若有代码从 cugraph_pyg.data 做 `from ... import TensorDictFeatureStore`
     再做 type annotation，在 mypy/pyright 下会报「TensorDictFeatureStore is not a type」；
     上游此 trade-off 已知，废弃期可接受。
   - DEPRECATED__TensorDictFeatureStore 别名前缀语义是「勿直接使用」，
     但 Python 无强制私有；DeprecationGate 通过 _wrapped_cls 替代
     module 级别的 DEPRECATED__ 裸暴露，信息封装更好。
   - 多 FutureWarning 触发（每次调用）：Python warning filter 默认 once/location，
     若用户在循环里构造 TensorDictFeatureStore，只看到一次警告；
     DeprecationGate.call_count 统计实际被调用次数，WALPURGIS_DEBUG 下可见。

Walpurgis 改写 20%（鲁迅拿法）:
- DeprecationGate: 替代裸 wrapper 函数，封装 (wrapped_cls, warning_msg, call_count)，
  __call__ 保持 wrapper 语义，额外统计调用次数 + DEBUG 打印；
  与上游函数完全等价，任意位置替换
- DeprecationPolicy: 替代 __init__.py 散落的多个 wrapper 函数，
  统一注册废弃对象，register(name, cls, msg) 一行，避免重复 warnings.warn 模板；
  policy.get(name) 返回 DeprecationGate（callable），行为等同上游 wrapper 函数
- InstanceCheckGuard: isinstance(obj, TensorDictFeatureStore) 的替代，
  isinstance(obj, InstanceCheckGuard(TensorDictFeatureStore)) 安全透传；
  上游 wrapper 函数化破坏了 isinstance，本模块提供修补路径
- 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖:
  断点1: DeprecationGate.__call__ 入口（参数摘要）
  断点2: warnings.warn 触发时机（call_count）
  断点3: 构造完成（返回对象类型）
  断点4: DeprecationPolicy.register（注册事件）
  断点5: DeprecationPolicy.get（查找事件）
  断点6: DeprecationPolicy.call_all_summary（批量调用统计）

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Type

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS feature_store_deprecation]", *args,
              file=sys.stderr, flush=True, **kwargs)


def _summarize_args(args, kwargs) -> str:
    """
    生成参数类型摘要字符串（不打印值，避免泄露数据/权重内容）。

    示例: args=(ndarray[100,128], int) kwargs={device='cuda', dtype=float32}
    """
    parts = []
    for a in args:
        t = type(a).__name__
        try:
            import torch
            if isinstance(a, torch.Tensor):
                parts.append(f"Tensor{list(a.shape)}")
                continue
        except ImportError:
            pass
        try:
            import numpy as np
            if isinstance(a, np.ndarray):
                parts.append(f"ndarray{list(a.shape)}")
                continue
        except ImportError:
            pass
        parts.append(t)
    kw_parts = [f"{k}={type(v).__name__}" for k, v in kwargs.items()]
    return f"args=({', '.join(parts)}) kwargs={{" + ", ".join(kw_parts) + "}"


# ──────────────────────────────────────────────────────────────────────────────
# DeprecationGate — 替代裸 wrapper 函数
# ──────────────────────────────────────────────────────────────────────────────

class DeprecationGate:
    """
    封装「废弃包装函数」的通用 callable 对象。

    上游 2d545b9 的 wrapper 函数（__init__.py）:

        def TensorDictFeatureStore(*args, **kwargs):
            warnings.warn(
                "TensorDictFeatureStore is deprecated. ...",
                FutureWarning,
            )
            return DEPRECATED__TensorDictFeatureStore(*args, **kwargs)

    Walpurgis 改写:
    - _wrapped_cls: 被包装的原始类（等同上游 DEPRECATED__TensorDictFeatureStore）
    - _warning_msg: FutureWarning 文字（与上游完全一致）
    - call_count: 累计调用次数（上游 wrapper 无此统计）
    - __call__: 与上游 wrapper 函数行为完全等价

    断点1: __call__ 入口，打印参数摘要
    断点2: warnings.warn 触发，打印 call_count
    断点3: 构造完成，打印返回对象类型
    """

    def __init__(self, wrapped_cls: type, warning_msg: str, stacklevel: int = 2):
        self._wrapped_cls = wrapped_cls
        self._warning_msg = warning_msg
        self._stacklevel = stacklevel
        self.call_count: int = 0
        # 让 repr 看起来像原始类名
        self.__name__ = wrapped_cls.__name__
        self.__qualname__ = wrapped_cls.__qualname__
        self.__module__ = wrapped_cls.__module__
        self.__doc__ = f"[DEPRECATED] {wrapped_cls.__doc__ or ''}"

    # ── 断点1: __call__ 入口 ──────────────────────────────────────────────
    def __call__(self, *args, **kwargs):
        """
        调用时触发 FutureWarning，然后透传构造原始类。
        行为与上游 wrapper 函数完全等价。
        """
        self.call_count += 1

        # ── 断点1 ────────────────────────────────────────────────────────
        _dbg(
            f"DeprecationGate.__call__(): "
            f"cls={self._wrapped_cls.__name__!r}, "
            f"call_count={self.call_count}, "
            f"{_summarize_args(args, kwargs)}"
        )

        # ── 断点2: warnings.warn ─────────────────────────────────────────
        _dbg(
            f"  触发 FutureWarning (call_count={self.call_count}): "
            f"{self._warning_msg[:80]}..."
        )
        warnings.warn(self._warning_msg, FutureWarning, stacklevel=self._stacklevel)

        result = self._wrapped_cls(*args, **kwargs)

        # ── 断点3: 构造完成 ──────────────────────────────────────────────
        _dbg(
            f"  构造完成: result_type={type(result).__name__}, "
            f"id=0x{id(result):016x}"
        )

        return result

    def __repr__(self):
        return (
            f"DeprecationGate("
            f"cls={self._wrapped_cls.__name__!r}, "
            f"call_count={self.call_count})"
        )

    def get_wrapped_class(self) -> type:
        """返回被包装的原始类（用于 isinstance 检查）。"""
        return self._wrapped_cls

    def reset_count(self) -> None:
        """重置调用计数（测试用）。"""
        self.call_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# InstanceCheckGuard — 修补 wrapper 函数化破坏 isinstance 的问题
# ──────────────────────────────────────────────────────────────────────────────

class InstanceCheckGuard:
    """
    上游 wrapper 函数化使 TensorDictFeatureStore 不再是 type，
    isinstance(obj, TensorDictFeatureStore) 会抛 TypeError。

    InstanceCheckGuard 提供安全透传:

        guard = InstanceCheckGuard(gate)
        isinstance(obj, guard)  →  isinstance(obj, DEPRECATED__TensorDictFeatureStore)

    用法:
        _guard = InstanceCheckGuard(TensorDictFeatureStore_gate)
        if isinstance(my_store, _guard):   # 不抛 TypeError
            ...

    注意: Python isinstance() 要求 arg2 实现 __instancecheck__ 的 metaclass
    或直接是 type；InstanceCheckGuard 通过 __class_getitem__ + metaclass 技巧实现，
    但更简洁的方式是在调用侧直接:
        isinstance(my_store, gate.get_wrapped_class())
    两种方式均在文档中说明。
    """

    def __init__(self, gate_or_cls):
        if isinstance(gate_or_cls, DeprecationGate):
            self._cls = gate_or_cls.get_wrapped_class()
        elif isinstance(gate_or_cls, type):
            self._cls = gate_or_cls
        else:
            raise TypeError(
                f"InstanceCheckGuard requires a DeprecationGate or type, "
                f"got {type(gate_or_cls).__name__}"
            )

    def __instancecheck__(self, instance) -> bool:
        return isinstance(instance, self._cls)

    def __repr__(self):
        return f"InstanceCheckGuard(cls={self._cls.__name__!r})"


# ──────────────────────────────────────────────────────────────────────────────
# DeprecationPolicy — 统一管理多个废弃对象
# ──────────────────────────────────────────────────────────────────────────────

class DeprecationPolicy:
    """
    统一注册和查找废弃对象，替代 __init__.py 中多个散落的 wrapper 函数。

    上游 __init__.py 模式（每个废弃类重复一遍）:
        def DaskGraphStore(*args, **kwargs):
            warnings.warn("... deprecated ...", FutureWarning)
            return DEPRECATED__DaskGraphStore(*args, **kwargs)

        def TensorDictFeatureStore(*args, **kwargs):
            warnings.warn("... deprecated ...", FutureWarning)
            return DEPRECATED__TensorDictFeatureStore(*args, **kwargs)

    Walpurgis 改写:
        policy = DeprecationPolicy()
        policy.register("TensorDictFeatureStore", cls, msg)
        policy.register("DaskGraphStore", cls, msg)
        gate = policy.get("TensorDictFeatureStore")  →  DeprecationGate
        store = gate(...)                             →  等同 wrapper 函数

    断点4: register() 事件
    断点5: get() 查找事件
    断点6: call_all_summary() 批量调用统计
    """

    def __init__(self):
        self._registry: Dict[str, DeprecationGate] = {}

    def register(
        self,
        name: str,
        wrapped_cls: type,
        warning_msg: str,
        stacklevel: int = 3,
    ) -> "DeprecationGate":
        """
        注册一个废弃对象，返回对应的 DeprecationGate。

        参数
        ----
        name        : 废弃对象的公开名称（如 \"TensorDictFeatureStore\"）
        wrapped_cls : 被包装的原始类（如 DEPRECATED__TensorDictFeatureStore）
        warning_msg : FutureWarning 文字（与上游保持一致）
        stacklevel  : warnings.warn stacklevel，默认 3（穿透 policy.get().__call__）

        返回
        ----
        DeprecationGate（可直接当 wrapper 函数用）
        """
        gate = DeprecationGate(wrapped_cls, warning_msg, stacklevel=stacklevel)
        self._registry[name] = gate

        # ── 断点4 ────────────────────────────────────────────────────────
        _dbg(
            f"DeprecationPolicy.register(): "
            f"name={name!r}, "
            f"cls={wrapped_cls.__name__!r}, "
            f"msg_prefix={warning_msg[:60]!r}..."
        )

        return gate

    def get(self, name: str) -> "DeprecationGate":
        """
        获取已注册的 DeprecationGate（callable，等同 wrapper 函数）。

        抛出
        ----
        KeyError  若 name 未注册
        """
        if name not in self._registry:
            raise KeyError(
                f"DeprecationPolicy: {name!r} 未注册。"
                f"已注册: {list(self._registry.keys())}"
            )

        gate = self._registry[name]

        # ── 断点5 ────────────────────────────────────────────────────────
        _dbg(
            f"DeprecationPolicy.get(): "
            f"name={name!r}, "
            f"gate={gate!r}"
        )

        return gate

    def has(self, name: str) -> bool:
        """查询 name 是否已注册。"""
        return name in self._registry

    def call_all_summary(self) -> Dict[str, int]:
        """
        返回所有已注册废弃对象的调用统计。

        返回
        ----
        dict: {name: call_count}

        断点6: 打印完整统计
        """
        summary = {name: gate.call_count for name, gate in self._registry.items()}

        # ── 断点6 ────────────────────────────────────────────────────────
        _dbg(f"DeprecationPolicy.call_all_summary(): {summary}")
        if _DEBUG:
            total = sum(summary.values())
            print(
                f"[WALPURGIS feature_store_deprecation] "
                f"废弃调用统计 (总计={total}):",
                file=sys.stderr,
            )
            for name, count in summary.items():
                bar = "#" * min(count, 40)
                print(
                    f"  {name:40s}: {count:6d} 次  {bar}",
                    file=sys.stderr,
                )

        return summary

    def __repr__(self):
        return (
            f"DeprecationPolicy("
            f"registered={list(self._registry.keys())})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 模块级废弃策略实例 — 对应上游 __init__.py 的两个 wrapper 函数
# ──────────────────────────────────────────────────────────────────────────────

#: 全局废弃策略实例（由 data/__init__.py 等导入使用）
_POLICY = DeprecationPolicy()

# 注册 TensorDictFeatureStore（commit 2d545b9 的核心变更）
try:
    from cugraph_pyg.data.feature_store import (
        TensorDictFeatureStore as _DEPRECATED__TensorDictFeatureStore,
        WholeFeatureStore,
    )

    #: TensorDictFeatureStore 废弃 gate — 等同上游 wrapper 函数
    #: 用法: store = TensorDictFeatureStore(data)  → FutureWarning + 原类实例
    TensorDictFeatureStore = _POLICY.register(
        "TensorDictFeatureStore",
        _DEPRECATED__TensorDictFeatureStore,
        (
            "TensorDictFeatureStore is deprecated.  Consider changing your "
            "workflow to launch using 'torchrun' and store data in "
            "the faster and more memory-efficient WholeFeatureStore instead."
        ),
    )

    _dbg(
        f"TensorDictFeatureStore 废弃 gate 注册成功: "
        f"{TensorDictFeatureStore!r}"
    )

except ImportError as _e:
    # cugraph-pyg 未安装（如纯 Walpurgis 环境）: 提供 stub，保持模块可导入
    _dbg(f"cugraph_pyg 未安装，TensorDictFeatureStore stub 模式: {_e}")

    class _StubClass:  # type: ignore[no-redef]
        """cugraph-pyg 未安装时的占位 stub。"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TensorDictFeatureStore requires cugraph-pyg. "
                "Install with: pip install cugraph-pyg"
            )

    TensorDictFeatureStore = _POLICY.register(
        "TensorDictFeatureStore",
        _StubClass,
        (
            "TensorDictFeatureStore is deprecated.  Consider changing your "
            "workflow to launch using 'torchrun' and store data in "
            "the faster and more memory-efficient WholeFeatureStore instead."
        ),
    )
    WholeFeatureStore = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# migrate 1e91ed7: Remove DaskGraphStore / CuGraphStore wrappers
#
# 上游 1e91ed7 (cugraph-gnn) 在 release 25.06 正式删除了 cugraph_pyg.data.__init__ 中
# DaskGraphStore/CuGraphStore 的 wrapper 函数和 dask_graph_store 整个模块。
#
# Walpurgis 迁移语义:
#   - DaskGraphStore 与 CuGraphStore 两个废弃 wrapper 从 DeprecationPolicy 中彻底移除。
#   - 任何尝试通过 _POLICY 获取这两个名称的调用将得到 KeyError，
#     错误信息引导用户迁移到 GraphStore / walpurgis.core.unified_store.UnifiedStore。
#   - 与上游不同: walpurgis 用 _RemovedEntryGuard 替代直接缺失，
#     提供比 KeyError 更友好的错误信息 + WALPURGIS_DEBUG 断点。
#
# 20% 改写 (鲁迅拿法):
#   - _RemovedEntryGuard: callable stub，调用时抛 RuntimeError（比 KeyError 更语义准确）
#     + WALPURGIS_DEBUG 断点打印调用堆栈摘要
#   - DeprecationPolicy.mark_removed(): 注册「已彻底移除」条目，
#     统一管理 removed vs deprecated 两类状态
#   - 断点7: mark_removed 注册事件
#   - 断点8: _RemovedEntryGuard.__call__ 触发（调用已删除API时打印堆栈）
# ──────────────────────────────────────────────────────────────────────────────


class _RemovedEntryGuard:
    """
    已彻底移除的 API 占位符。

    上游 1e91ed7 直接删除了 DaskGraphStore/CuGraphStore wrapper；
    Walpurgis 保留一个可检测的对象，使得 `from walpurgis.core... import DaskGraphStore`
    不会 ImportError（模块可正常加载），但实际调用时抛出带迁移指引的 RuntimeError。

    断点8: 调用时打印调用堆栈摘要。
    """

    def __init__(self, removed_name: str, migration_hint: str):
        self._name = removed_name
        self._hint = migration_hint
        self.__name__ = removed_name
        self.__qualname__ = removed_name

    def __call__(self, *args, **kwargs):
        import traceback

        # ── 断点8 ────────────────────────────────────────────────────────
        _dbg(
            f"_RemovedEntryGuard.__call__(): {self._name!r} 已彻底移除，"
            f"调用者: {''.join(traceback.format_stack()[-4:-1]).strip()[:200]}"
        )

        raise RuntimeError(
            f"{self._name} has been removed (upstream cugraph-gnn 1e91ed7, release 25.06).\n"
            f"Migration: {self._hint}"
        )

    def __repr__(self):
        return f"_RemovedEntryGuard(name={self._name!r})"


def _add_mark_removed(policy_cls):
    """
    为 DeprecationPolicy 类动态添加 mark_removed() 方法。
    避免修改类定义区（保持已有迁移条目的可读性）。

    断点7: mark_removed 注册事件。
    """

    def mark_removed(self, name: str, migration_hint: str) -> "_RemovedEntryGuard":
        """
        注册「已彻底移除」条目。

        与 register() 不同：mark_removed 的条目调用时抛 RuntimeError 而非 FutureWarning。
        用于 upstream 已删除（非废弃中）的 API。
        """
        guard = _RemovedEntryGuard(name, migration_hint)
        self._registry[name] = guard  # 复用同一 registry，统一 has() / 遍历

        # ── 断点7 ────────────────────────────────────────────────────────
        _dbg(
            f"DeprecationPolicy.mark_removed(): "
            f"name={name!r}, hint_prefix={migration_hint[:60]!r}..."
        )

        return guard

    policy_cls.mark_removed = mark_removed


_add_mark_removed(DeprecationPolicy)


# 移除 DaskGraphStore 和 CuGraphStore (对应上游 1e91ed7 删除 dask_graph_store.py)
DaskGraphStore = _POLICY.mark_removed(
    "DaskGraphStore",
    "Use walpurgis.core.unified_store.UnifiedStore (or cugraph_pyg.data.GraphStore) instead."
    " The entire Dask-based distributed graph API was dropped in release 25.06.",
)

CuGraphStore = _POLICY.mark_removed(
    "CuGraphStore",
    "CuGraphStore was renamed to DaskGraphStore, which is now also removed."
    " Use walpurgis.core.unified_store.UnifiedStore instead.",
)

_dbg(
    f"1e91ed7 迁移完成: DaskGraphStore/CuGraphStore 已注册为 _RemovedEntryGuard, "
    f"policy registry keys={list(_POLICY._registry.keys())}"
)


# ──────────────────────────────────────────────────────────────────────────────
# 便捷导出
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DeprecationGate",
    "DeprecationPolicy",
    "InstanceCheckGuard",
    "TensorDictFeatureStore",
    "WholeFeatureStore",
    "DaskGraphStore",
    "CuGraphStore",
    "_RemovedEntryGuard",
    "_POLICY",
]
