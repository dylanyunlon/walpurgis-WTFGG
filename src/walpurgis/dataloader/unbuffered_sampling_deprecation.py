"""
unbuffered_sampling_deprecation.py — 8074120 迁移: 废弃 unbuffered sampling (disk-based)

migrate 8074120: Deprecate Unbuffered Sampling in cuGraph-PyG (#151)

上游变化 (8074120, cugraph-gnn, alexbarghi-nv, 2025-03-03, PR #151):
    4 files changed, 15 insertions(+), 195 deletions(-)

1. 删除 examples/cugraph_dist_sampling_mg.py (112行) — 多 GPU unbuffered 采样示例
2. 删除 examples/cugraph_dist_sampling_sg.py  (82行)  — 单 GPU unbuffered 采样示例
3. loader/link_neighbor_loader.py — LinkNeighborLoader.__init__ 加 FutureWarning:
       if directory is not None:
           warnings.warn(
               "Unbuffered sampling, where samples are dumped to disk"
               ", is deprecated in cuGraph-PyG and will be removed in release 25.06.",
               FutureWarning,
           )
4. loader/neighbor_loader.py — NeighborLoader.__init__ 加相同 FutureWarning;
       另: directory 参数注释从无到 "# Deprecated."

CI/merge/examples → SKIP:
- examples/cugraph_dist_sampling_mg.py — SKIP: 示例文件，Walpurgis 无对应多 GPU 示例体系
- examples/cugraph_dist_sampling_sg.py — SKIP: 示例文件，Walpurgis 无对应单 GPU 示例体系

迁移位置: src/walpurgis/dataloader/unbuffered_sampling_deprecation.py — 新建
    loader/link_neighbor_loader.py + loader/neighbor_loader.py 的 directory 废弃逻辑
    在 Walpurgis 中统一归到本模块，与 loader_deprecation.py 并列。

鲁迅拿法改写（≥20%）:
1. UnbufferedSamplingMode enum — 上游只有 `directory is not None` 的布尔判断；
   Walpurgis 显式枚举 BUFFERED/UNBUFFERED 两种模式，模式切换可被拦截和审计。
2. UnbufferedSamplingPolicy dataclass — 封装废弃策略（目标版本/警告消息/是否 raise），
   上游硬编码字符串 "25.06"；此处字段化，方便后续版本更新时单点修改。
3. UnbufferedSamplingGuard — 守卫对象，`check(directory)` 统一触发 FutureWarning；
   上游在 LinkNeighborLoader / NeighborLoader 各自独立写 `if directory is not None`，
   代码重复且无可观测性；本类消除重复并加 call_count + 断点。
4. DirectoryArgAudit dataclass — 记录每次 `directory` 参数使用情况（loader 类名/调用次数），
   上游无任何审计记录；本 dataclass 使废弃用量可量化，辅助评估何时可以删除。
5. 全链路 WALPURGIS_DEBUG=1 断点（5处）

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    """WALPURGIS_DEBUG=1 断点打印。所有 kv 只打印类型或简短值，不泄露数据内容。"""
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v!r}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. UnbufferedSamplingMode
#    上游: `if directory is not None` — 隐式布尔，无名称，无文档
#    Walpurgis: 显式 enum，BUFFERED = 内存 streaming；UNBUFFERED = 磁盘 dump (废弃)
# ─────────────────────────────────────────────────────────────────────────────

class UnbufferedSamplingMode(Enum):
    """
    采样模式枚举，对应 directory 参数的有/无。

    BUFFERED   — directory=None，样本留在内存；25.06 后唯一受支持模式。
    UNBUFFERED — directory=<path>，样本 dump 到磁盘；8074120 起废弃，25.06 移除。
    """
    BUFFERED = auto()
    UNBUFFERED = auto()  # Deprecated since 8074120

    @classmethod
    def from_directory(cls, directory: Optional[str]) -> "UnbufferedSamplingMode":
        """
        断点1: 从 directory 参数推断模式，打印推断结果。

        等价于上游的 `if directory is not None` 判断，但给模式命名，
        使调用方意图显式可读。
        """
        mode = cls.UNBUFFERED if directory is not None else cls.BUFFERED
        _dbg(
            "UnbufferedSamplingMode.from_directory",
            f"directory={'<set>' if directory else 'None'} → mode={mode.name}",
            directory_type=type(directory).__name__,
        )
        return mode

    @property
    def is_deprecated(self) -> bool:
        return self is UnbufferedSamplingMode.UNBUFFERED


# ─────────────────────────────────────────────────────────────────────────────
# 2. UnbufferedSamplingPolicy
#    上游: 硬编码 "25.06" + 固定字符串；Walpurgis: 字段化，单点可改
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnbufferedSamplingPolicy:
    """
    废弃策略配置。

    上游 8074120 在两个 loader 文件里各硬编码了:
        "Unbuffered sampling, where samples are dumped to disk"
        ", is deprecated in cuGraph-PyG and will be removed in release 25.06."
    Walpurgis 将版本号、消息模板、严格模式字段化，后续版本更新只改这里。

    strict=True 时 check() 改为 raise DeprecationWarning（用于 CI 测试环境）。
    strict=False（默认）等同上游行为: 发 FutureWarning 但继续执行。
    """
    removal_version: str = "25.06"
    strict: bool = False

    @property
    def warning_message(self) -> str:
        return (
            "Unbuffered sampling, where samples are dumped to disk"
            f", is deprecated in cuGraph-PyG / Walpurgis"
            f" and will be removed in release {self.removal_version}."
        )

    @property
    def strict_message(self) -> str:
        return (
            f"[strict mode] Unbuffered sampling (directory != None) is rejected;"
            f" scheduled for removal in {self.removal_version}."
        )

    def emit(self, stacklevel: int = 3) -> None:
        """
        断点2: 发出废弃警告。strict=True 时改为 raise。

        stacklevel 默认 3，使 warnings 指向调用方的调用方（loader.__init__ 的调用者），
        与上游行为一致。
        """
        _dbg(
            "UnbufferedSamplingPolicy.emit",
            f"strict={self.strict}, removal_version={self.removal_version}",
            stacklevel=stacklevel,
        )
        if self.strict:
            raise DeprecationWarning(self.strict_message)
        warnings.warn(self.warning_message, FutureWarning, stacklevel=stacklevel)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DirectoryArgAudit
#    上游: 零审计；Walpurgis: 按 loader 类名记录调用次数，可量化废弃用量
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DirectoryArgAudit:
    """
    记录各 loader 类使用 directory 参数（unbuffered 模式）的次数。

    上游 8074120 只发警告，没有任何量化手段。
    此 dataclass 使评估「何时可以安全删除 directory 支持」有数据依据。

    字段:
        counts — {loader_class_name: call_count}
        total  — 所有 loader 的合计调用次数
    """
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def record(self, loader_name: str) -> None:
        """
        断点3: 记录一次 directory 参数使用。

        loader_name 典型值: "NeighborLoader" / "LinkNeighborLoader"。
        """
        self.counts[loader_name] = self.counts.get(loader_name, 0) + 1
        _dbg(
            "DirectoryArgAudit.record",
            f"loader={loader_name!r}, count={self.counts[loader_name]}, total={self.total}",
        )

    def describe(self) -> str:
        """生成可读的审计报告，供日志或 CI summary 使用。"""
        if not self.counts:
            return "DirectoryArgAudit: no unbuffered sampling calls recorded."
        lines = [f"DirectoryArgAudit (total={self.total}):"]
        for name, cnt in sorted(self.counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {cnt} call(s)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DirectoryArgAudit(total={self.total}, loaders={list(self.counts.keys())})"


# ─────────────────────────────────────────────────────────────────────────────
# 4. UnbufferedSamplingGuard
#    上游: 两个 loader 各写一段 `if directory is not None: warnings.warn(...)`
#    Walpurgis: 单一 Guard 消除重复，含 audit + 断点
# ─────────────────────────────────────────────────────────────────────────────

class UnbufferedSamplingGuard:
    """
    统一守卫，替代上游在 NeighborLoader / LinkNeighborLoader 各自独立的废弃检查。

    上游 8074120 实现模式 (重复两次):
        if directory is not None:
            warnings.warn(
                "Unbuffered sampling, where samples are dumped to disk"
                ", is deprecated in cuGraph-PyG and will be removed in release 25.06.",
                FutureWarning,
            )

    UnbufferedSamplingGuard.check(directory, loader_name) 等价但:
    - 消除两文件重复代码
    - 通过 DirectoryArgAudit 记录每次调用
    - 通过 UnbufferedSamplingMode.from_directory 将判断语义化
    - WALPURGIS_DEBUG=1 时打印完整上下文

    断点4: check() — 打印 loader_name / mode / directory 类型
    """

    def __init__(
        self,
        policy: Optional[UnbufferedSamplingPolicy] = None,
        audit: Optional[DirectoryArgAudit] = None,
    ):
        self._policy = policy or UnbufferedSamplingPolicy()
        self._audit = audit or DirectoryArgAudit()

    @property
    def audit(self) -> DirectoryArgAudit:
        return self._audit

    @property
    def policy(self) -> UnbufferedSamplingPolicy:
        return self._policy

    def check(self, directory: Optional[str], loader_name: str = "Loader") -> UnbufferedSamplingMode:
        """
        检查 directory 参数；如为 unbuffered 模式，触发废弃警告并记录审计。

        返回当前采样模式枚举，供调用方做后续分支判断。

        断点4: 打印 loader_name / 推断模式 / directory 是否为 None
        """
        mode = UnbufferedSamplingMode.from_directory(directory)

        _dbg(
            "UnbufferedSamplingGuard.check",
            f"loader={loader_name!r}, mode={mode.name}",
            directory_is_none=(directory is None),
        )

        if mode.is_deprecated:
            self._audit.record(loader_name)
            # stacklevel=4: Guard.check → loader.__init__ → 用户代码
            self._policy.emit(stacklevel=4)

        return mode

    def summary(self) -> str:
        """
        断点5: 打印审计汇总，供 teardown / CI 日志使用。

        等价于上游无对应功能（上游只发警告，不汇总）。
        """
        report = self._audit.describe()
        _dbg("UnbufferedSamplingGuard.summary", report)
        return report

    def __repr__(self) -> str:
        return (
            f"UnbufferedSamplingGuard("
            f"policy={self._policy!r}, "
            f"audit={self._audit!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例 — 供 dataloader/ 中各 loader 直接调用
#
# 用法（在 NeighborLoader.__init__ / LinkNeighborLoader.__init__ 中）:
#
#     from walpurgis.dataloader.unbuffered_sampling_deprecation import (
#         unbuffered_sampling_guard,
#     )
#     ...
#     unbuffered_sampling_guard.check(directory, loader_name="NeighborLoader")
#
# 替代上游两处独立的:
#     if directory is not None:
#         warnings.warn("...", FutureWarning)
# ─────────────────────────────────────────────────────────────────────────────

unbuffered_sampling_guard: UnbufferedSamplingGuard = UnbufferedSamplingGuard()


# ─────────────────────────────────────────────────────────────────────────────
# 自测 (python -c "exec(open('...').read())")
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import warnings as _w

    passed = 0

    # T1: BUFFERED 模式 — directory=None，不触发警告
    mode = UnbufferedSamplingMode.from_directory(None)
    assert mode is UnbufferedSamplingMode.BUFFERED, "T1 failed"
    assert not mode.is_deprecated, "T1b failed"
    passed += 1; print("[PASS] T1 BUFFERED mode from directory=None")

    # T2: UNBUFFERED 模式 — directory=path，触发废弃
    mode = UnbufferedSamplingMode.from_directory("/tmp/samples")
    assert mode is UnbufferedSamplingMode.UNBUFFERED, "T2 failed"
    assert mode.is_deprecated, "T2b failed"
    passed += 1; print("[PASS] T2 UNBUFFERED mode from directory=<path>")

    # T3: UnbufferedSamplingPolicy.emit 发出 FutureWarning
    policy = UnbufferedSamplingPolicy(removal_version="25.06", strict=False)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        policy.emit(stacklevel=1)
    assert len(caught) == 1, f"T3 failed: expected 1 warning, got {len(caught)}"
    assert issubclass(caught[0].category, FutureWarning), "T3b failed"
    assert "25.06" in str(caught[0].message), "T3c failed"
    passed += 1; print("[PASS] T3 policy.emit FutureWarning with removal_version")

    # T4: UnbufferedSamplingPolicy strict=True → raise DeprecationWarning
    strict_policy = UnbufferedSamplingPolicy(strict=True)
    try:
        strict_policy.emit()
        assert False, "T4 failed: should have raised"
    except DeprecationWarning:
        pass
    passed += 1; print("[PASS] T4 strict policy raises DeprecationWarning")

    # T5: DirectoryArgAudit.record 累计计数
    audit = DirectoryArgAudit()
    audit.record("NeighborLoader")
    audit.record("NeighborLoader")
    audit.record("LinkNeighborLoader")
    assert audit.total == 3, f"T5 failed: total={audit.total}"
    assert audit.counts["NeighborLoader"] == 2, "T5b failed"
    assert audit.counts["LinkNeighborLoader"] == 1, "T5c failed"
    passed += 1; print("[PASS] T5 DirectoryArgAudit records per-loader counts")

    # T6: DirectoryArgAudit.describe 输出包含 loader 名
    desc = audit.describe()
    assert "NeighborLoader" in desc, "T6 failed"
    assert "total=3" in desc, "T6b failed"
    passed += 1; print("[PASS] T6 DirectoryArgAudit.describe output")

    # T7: UnbufferedSamplingGuard.check — buffered path, no warning
    guard = UnbufferedSamplingGuard()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = guard.check(None, loader_name="NeighborLoader")
    assert result is UnbufferedSamplingMode.BUFFERED, "T7 failed"
    assert len(caught) == 0, f"T7b failed: unexpected warning {caught}"
    assert guard.audit.total == 0, "T7c failed"
    passed += 1; print("[PASS] T7 guard.check(None) — no warning, no audit record")

    # T8: UnbufferedSamplingGuard.check — unbuffered path, warning + audit
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = guard.check("/tmp/s", loader_name="LinkNeighborLoader")
    assert result is UnbufferedSamplingMode.UNBUFFERED, "T8 failed"
    assert len(caught) == 1, f"T8b failed: expected 1 warning, got {len(caught)}"
    assert issubclass(caught[0].category, FutureWarning), "T8c failed"
    assert guard.audit.counts.get("LinkNeighborLoader") == 1, "T8d failed"
    passed += 1; print("[PASS] T8 guard.check(<path>) — FutureWarning + audit recorded")

    # T9: global unbuffered_sampling_guard 是 UnbufferedSamplingGuard 实例
    assert isinstance(unbuffered_sampling_guard, UnbufferedSamplingGuard), "T9 failed"
    passed += 1; print("[PASS] T9 global unbuffered_sampling_guard instance")

    # T10: guard.summary() 包含 audit 信息
    summary = guard.summary()
    assert "LinkNeighborLoader" in summary, "T10 failed"
    passed += 1; print("[PASS] T10 guard.summary() includes audit data")

    print(f"\n全部自测通过: {passed}/10 [PASS]")


if __name__ == "__main__":
    _self_test()
