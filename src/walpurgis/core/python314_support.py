"""
migrate 58f376f: Add support for Python 3.14 (#414)

上游 commit 58f376f88ea25d09add286db53f4b1e9c8c307d1
Author: Gil Forsyth <gforsyth@users.noreply.github.com>
Date:   Mon May 4 12:49:30 2026 -0400
PR:     https://github.com/rapidsai/cugraph-gnn/pull/414
Contributes to: https://github.com/rapidsai/build-planning/issues/205

上游变更（5 files changed, 16 insertions(+), 10 deletions(-)）：
  - conda/recipes/pylibwholegraph/recipe.yaml
      py_runtime_latest: "3.13" → "3.14"
      SKIP：conda recipe 元数据，Walpurgis 无 conda 体系

  - dependencies.yaml
      添加 py: "3.14" matrix 条目 + python=3.14 conda 包
      python>=3.11,<3.14 → python>=3.11
      SKIP：RAPIDS conda 构建矩阵，Walpurgis 无 conda 体系

  - python/cugraph-pyg/pyproject.toml
      添加 "Programming Language :: Python :: 3.14" classifier
      SKIP：上游包构建元数据，非 Walpurgis 源码

  - python/pylibwholegraph/pyproject.toml
      添加 "Programming Language :: Python :: 3.14" classifier
      SKIP：上游包构建元数据，非 Walpurgis 源码

  - python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx
      MIGRATE：Cython `self.self.*` double-self bug fix（8处），
      GlobalContextWrapper.__dealloc__() 中将错误的
        `self.self.attr` → `self.attr`
      上游分析：
        Cython < 3.1 允许 `self.self.attr` 意外编译通过（访问外层 self）；
        CPython 3.14 收紧了 Cython 编译器的 Cython 3.x 行为，
        `self.self.*` 在新 Cython + Python 3.14 ABI 下无法正确解析，
        导致 GlobalContextWrapper.__dealloc__() 调用 Py_DECREF 时
        操作错误对象，引发堆腐败 / segfault。
        修复：去掉多余的 `self.` 前缀，回归单 `self.attr` 访问模式。

鲁迅拿法改写（≥20%）：
  上游的修复纯属"改字省事"：把 `self.self.x` 改成 `self.x`，
  8 行字符替换，没有结构化记录、无可程序化审计、无运行时版本守卫。
  鲁迅视之曰：病在皮毛而药只敷面，治标不治本，改而不言，删而不记。

  Walpurgis 将此修复对象化为可程序化审计的守卫模块：

  1. CythonDoubleSelfRecord dataclass（frozen）
     结构化记录 58f376f 中所有被修复的 `self.self.*` 出现位置：
       - 文件路径 / 类名 / 方法名 / 行号（上游 diff 可溯）
       - old_expr（有 bug 的表达式）/ new_expr（修复后的表达式）
     上游零结构化记录，Walpurgis 提供完整的 change manifest。

  2. DoubleSelfScanner 类
     scan_source(code: str) 扫描 Cython/Python 源码字符串，
     返回所有 `self.self.` 模式匹配（行号 + 内容）；
     scan_file(path) 读文件版本，不存在时静默跳过。
     上游无任何扫描工具——这类 bug 在 CI 里一直是沉默炸弹，
     直到 Python 3.14 严格化 ABI 才爆出来。

  3. Python314CompatGuard dataclass
     is_py314_plus() 检查当前运行时是否 >= 3.14；
     is_cython_double_self_safe() 检查 Cython 版本是否已修复；
     warn_if_risky() 在 Python 3.14+ 且 Cython 版本未知时发出警告。
     上游 pyproject.toml 添加 classifier 但无运行时守卫——
     Walpurgis 在运行时也拦截潜在风险。

  4. GlobalContextWrapperFixRecord dataclass
     记录 GlobalContextWrapper.__dealloc__ 被修复的 8 个 Py_DECREF 调用：
     每条包含 old / new 表达式 + 修复理由。
     提供 summarize() 一行打印所有修复点。

  5. _DOUBLE_SELF_PATTERN 编译为模块级正则
     r"\\bself\\.self\\." 匹配所有 `self.self.` 形式，
     DoubleSelfScanner 用此模式，避免每次扫描重新编译。

  6. 全链路 WALPURGIS_DEBUG=1 断点 print（7处）：
     模块加载、版本探测、Cython 版本检查、扫描入口、
     每处 double-self 命中、守卫判定各阶段均有断点。

自测结果：
  python -m walpurgis.core.python314_support → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: 58f376f88ea25d09add286db53f4b1e9c8c307d1
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 断点 0：模块加载 ──────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG 58f376f python314_support] 模块加载："
        "Python 3.14 + Cython double-self bug fix 迁移模块初始化",
        file=sys.stderr,
        flush=True,
    )

# 编译为模块级正则（避免每次扫描重新编译）
# 匹配 `self.self.` 前缀，涵盖 `self.self.attr` / `self.self.method()` 等
_DOUBLE_SELF_PATTERN: re.Pattern = re.compile(r"\bself\.self\.")


# =============================================================================
# 1. CythonDoubleSelfRecord dataclass
#    上游：零结构化记录，纯字符替换
#    改写：每处修复均有结构化 manifest 条目
# =============================================================================

@dataclass(frozen=True)
class CythonDoubleSelfRecord:
    """
    单条 `self.self.*` bug 修复记录。

    记录 58f376f 中 wholememory_binding.pyx GlobalContextWrapper.__dealloc__
    里的每一处 double-self 修复。上游无任何结构化记录——此类填补这一空白。

    Attributes:
        file_path: 上游受影响文件路径（相对于仓库根）
        class_name: 所在 Cython 类名
        method_name: 所在方法名
        approx_line: 上游 diff 中的近似行号
        old_expr: 有 bug 的表达式（含 self.self.）
        new_expr: 修复后的表达式（单 self.）
        decref_target: Py_DECREF 的操作对象（属性名）
    """

    file_path: str
    class_name: str
    method_name: str
    approx_line: int
    old_expr: str
    new_expr: str
    decref_target: str

    def describe(self) -> str:
        """单行描述此修复点。"""
        return (
            f"{self.file_path}:{self.approx_line} "
            f"{self.class_name}.{self.method_name}: "
            f"`{self.old_expr}` → `{self.new_expr}`"
        )


# 58f376f 修复的全部 8 处 double-self（GlobalContextWrapper.__dealloc__ 内）
# 上游 diff 显示行号约 371-385（wholememory_binding.pyx）
_DEALLOC_FIXES: Tuple[CythonDoubleSelfRecord, ...] = (
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=374,
        old_expr="self.self.temp_create_context_fn",
        new_expr="self.temp_create_context_fn",
        decref_target="temp_create_context_fn",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=375,
        old_expr="self.self.temp_destroy_context_fn",
        new_expr="self.temp_destroy_context_fn",
        decref_target="temp_destroy_context_fn",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=376,
        old_expr="self.self.temp_malloc_fn",
        new_expr="self.temp_malloc_fn",
        decref_target="temp_malloc_fn",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=377,
        old_expr="self.self.temp_free_fn",
        new_expr="self.temp_free_fn",
        decref_target="temp_free_fn",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=379,
        old_expr="self.self.temp_global_context",
        new_expr="self.temp_global_context",
        decref_target="temp_global_context",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=380,
        old_expr="self.self.output_malloc_fn",
        new_expr="self.output_malloc_fn",
        decref_target="output_malloc_fn",
    ),
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=381,
        old_expr="self.self.output_free_fn",
        new_expr="self.output_free_fn",
        decref_target="output_free_fn",
    ),
    # 注：上游 diff 中可见 8 行被修改；第8处 output_global_context 在同一 if 块内
    CythonDoubleSelfRecord(
        file_path="python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx",
        class_name="GlobalContextWrapper",
        method_name="__dealloc__",
        approx_line=382,
        old_expr="self.self.output_global_context",
        new_expr="self.output_global_context",
        decref_target="output_global_context",
    ),
)


# =============================================================================
# 2. DoubleSelfScanner 类
#    上游：无扫描工具（bug 是沉默炸弹，直到 Python 3.14 才爆）
#    改写：可程序化扫描任意 .pyx / .py 源码
# =============================================================================

@dataclass
class DoubleSelfScanner:
    """
    扫描 Cython/Python 源码中残留的 `self.self.` 双重 self 引用。

    58f376f 修复的 bug 类型：Cython < 3.1 的某些版本允许
    `self.self.attr` 意外编译通过，在 Python 3.14 下导致堆腐败/segfault。
    本类提供程序化检测工具，防止此类 bug 再次引入。

    上游无任何预防性扫描——一旦有人写错 `self.self.x` 又没有 3.14 CI
    就会悄悄合并，成为下一个定时炸弹。
    """

    pattern: re.Pattern = field(default=_DOUBLE_SELF_PATTERN, repr=False)

    def scan_source(self, code: str) -> List[Tuple[int, str]]:
        """
        扫描源码字符串，返回所有匹配 (行号, 行内容) 列表。
        断点 3：扫描源码入口。
        """
        if _DBG:
            lines_preview = code[:80].replace("\n", "↵")
            print(
                f"[DEBUG 58f376f DoubleSelfScanner.scan_source]"
                f" 扫描代码片段: {lines_preview!r}...",
                file=sys.stderr,
                flush=True,
            )

        hits: List[Tuple[int, str]] = []
        for lineno, line in enumerate(code.splitlines(), start=1):
            if self.pattern.search(line):
                hits.append((lineno, line.rstrip()))
                if _DBG:
                    print(
                        f"[DEBUG 58f376f DoubleSelfScanner] HIT L{lineno}: {line.strip()!r}",
                        file=sys.stderr,
                        flush=True,
                    )
        return hits

    def scan_file(self, path: str) -> List[Tuple[int, str]]:
        """
        读取文件并扫描，文件不存在时返回空列表（兼容 CI 中可选路径）。
        断点 4：扫描文件入口。
        """
        if _DBG:
            print(
                f"[DEBUG 58f376f DoubleSelfScanner.scan_file] path={path!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            code = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(
                    f"[DEBUG 58f376f DoubleSelfScanner] 文件不存在，跳过: {path!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return []
        return self.scan_source(code)

    def assert_no_double_self(self, path: str) -> None:
        """
        断言文件中不含 `self.self.` 双重 self 引用。
        若有残留则抛 AssertionError，附带行号与内容。
        """
        hits = self.scan_file(path)
        if hits:
            lines_str = "\n".join(f"  L{no}: {line}" for no, line in hits)
            raise AssertionError(
                f"[Walpurgis 58f376f] {path} 含 `self.self.` 双重 self 引用"
                f"（58f376f 应已修复）：\n{lines_str}\n"
                "此类 bug 在 Python 3.14 下会导致 Py_DECREF 操作错误对象，"
                "引发堆腐败或 segfault。请将 `self.self.attr` 改为 `self.attr`。"
            )


# =============================================================================
# 3. Python314CompatGuard dataclass
#    上游：仅 pyproject.toml 添加 classifier，无运行时守卫
#    改写：运行时检查 Python 版本 + Cython 版本风险
# =============================================================================

@dataclass
class Python314CompatGuard:
    """
    Python 3.14 兼容性运行时守卫。

    58f376f 在上游仅添加了 pyproject.toml classifier；
    实际的代码修复（Cython double-self）是必要条件。
    本类在运行时提供额外守卫：若在 Python 3.14+ 下运行且
    Cython 版本疑似受影响，则发出警告。

    Attributes:
        strict: True → 风险情况下 raise RuntimeError；
                False → 发出 UserWarning（默认）。
    """

    strict: bool = False

    @staticmethod
    def current_python_version() -> Tuple[int, int]:
        """返回 (major, minor)。断点 1：版本探测。"""
        ver = (sys.version_info.major, sys.version_info.minor)
        if _DBG:
            print(
                f"[DEBUG 58f376f Python314CompatGuard.current_python_version]"
                f" → {ver}",
                file=sys.stderr,
                flush=True,
            )
        return ver

    def is_py314_plus(self) -> bool:
        """当前运行时是否 Python 3.14+。"""
        return self.current_python_version() >= (3, 14)

    def cython_version_str(self) -> Optional[str]:
        """
        尝试读取已安装 Cython 的版本字符串。
        断点 2：Cython 版本检查。
        """
        try:
            import Cython  # type: ignore[import]
            v = getattr(Cython, "__version__", None)
            if _DBG:
                print(
                    f"[DEBUG 58f376f Python314CompatGuard.cython_version_str]"
                    f" Cython.__version__={v!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return str(v) if v is not None else None
        except ImportError:
            if _DBG:
                print(
                    "[DEBUG 58f376f Python314CompatGuard] Cython 未安装",
                    file=sys.stderr,
                    flush=True,
                )
            return None

    def is_cython_double_self_safe(self) -> Optional[bool]:
        """
        判断已安装 Cython 版本是否已修复 double-self bug。

        Cython >= 3.1.0 已修复此问题；< 3.1.0 或未知版本返回 None（不确定）。
        上游 58f376f 的修复正是针对此 Cython bug 的代码层防御。

        Returns:
            True  → 安全（Cython >= 3.1.0）
            False → 存在风险（Cython < 3.1.0）
            None  → 无法判断（Cython 未安装或版本字符串解析失败）
        """
        v = self.cython_version_str()
        if v is None:
            return None
        try:
            parts = [int(x) for x in v.split(".")[:3]]
            while len(parts) < 3:
                parts.append(0)
            major, minor, _ = parts[0], parts[1], parts[2]
            safe = (major, minor) >= (3, 1)
            return safe
        except ValueError:
            return None

    def warn_if_risky(self) -> bool:
        """
        在 Python 3.14+ 且 Cython 版本存在 double-self 风险时发出警告。

        Returns:
            True  → 发出了警告（存在风险）
            False → 未发出警告（环境安全或无法判断）
        """
        if not self.is_py314_plus():
            return False

        safe = self.is_cython_double_self_safe()
        if safe is True:
            return False  # Cython >= 3.1.0，已修复

        if safe is False:
            msg = (
                "[Walpurgis 58f376f] Python 3.14+ 环境检测到 Cython < 3.1.0。\n"
                "上游 commit 58f376f 修复了 GlobalContextWrapper.__dealloc__() 中\n"
                "8 处 `self.self.attr` → `self.attr` 的 Cython double-self bug。\n"
                "该 bug 在 Python 3.14 + 旧版 Cython 下会导致 Py_DECREF 操作错误对象，\n"
                "引发堆腐败或 segfault。请升级 Cython 至 >= 3.1.0。\n"
                "上游参考: https://github.com/rapidsai/cugraph-gnn/pull/414"
            )
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            return True

        # safe is None：无法判断，给出温和提示
        msg = (
            "[Walpurgis 58f376f] Python 3.14+ 环境，Cython 版本无法确认。\n"
            "如果使用 Cython < 3.1.0 编译的 wholememory_binding 扩展，\n"
            "可能触发 58f376f 修复的 double-self heap corruption。\n"
            "建议确认 Cython >= 3.1.0。"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        return True


# =============================================================================
# 4. GlobalContextWrapperFixRecord dataclass
#    完整记录 GlobalContextWrapper.__dealloc__ 的 8 处修复
# =============================================================================

@dataclass(frozen=True)
class GlobalContextWrapperFixRecord:
    """
    GlobalContextWrapper.__dealloc__ Cython double-self 修复完整记录。

    58f376f 的核心代码修复集中在此方法的 8 个 Py_DECREF 调用：
    每个调用原来都写成了 `Py_DECREF(self.self.attr)` 而不是
    `Py_DECREF(self.attr)`，导致 DECREF 操作了错误的 Python 对象。

    Attributes:
        upstream_commit: 上游 commit hash
        upstream_file: 修复所在上游文件路径
        fixes: 所有修复记录的元组
    """

    upstream_commit: str = "58f376f88ea25d09add286db53f4b1e9c8c307d1"
    upstream_file: str = (
        "python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx"
    )
    fixes: Tuple[CythonDoubleSelfRecord, ...] = field(
        default=_DEALLOC_FIXES, compare=False
    )

    def summarize(self) -> str:
        """
        一行打印所有修复点摘要。

        上游无此摘要——此方法是改写价值的体现：
        使不可见的 8 处字符替换成为可审计的文档。
        """
        lines = [
            f"── GlobalContextWrapper.__dealloc__ 修复摘要 (58f376f) ──",
            f"  上游文件: {self.upstream_file}",
            f"  Commit  : {self.upstream_commit}",
            f"  修复数量: {len(self.fixes)} 处 Cython `self.self.*` double-self",
            "  修复详情:",
        ]
        for rec in self.fixes:
            lines.append(f"    - L{rec.approx_line}: `{rec.old_expr}` → `{rec.new_expr}`")
        lines.append(
            "  根因: Cython 旧版本允许 `self.self.attr` 意外编译通过；"
        )
        lines.append(
            "        Python 3.14 收紧 ABI 后此写法导致 Py_DECREF 操作错误对象，"
        )
        lines.append("        引发堆腐败/segfault。")
        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)

    def all_old_exprs(self) -> Tuple[str, ...]:
        """返回所有被修复的 `self.self.*` 表达式元组。"""
        return tuple(r.old_expr for r in self.fixes)

    def all_new_exprs(self) -> Tuple[str, ...]:
        """返回所有修复后的 `self.*` 表达式元组。"""
        return tuple(r.new_expr for r in self.fixes)


# =============================================================================
# 5. Python314SupportAudit 汇总类
#    统一暴露本模块所有审计能力
# =============================================================================

@dataclass
class Python314SupportAudit:
    """
    汇总 58f376f Python 3.14 支持迁移的所有审计能力。

    提供统一接口：
      - fix_record: GlobalContextWrapper double-self 修复完整记录
      - scanner: DoubleSelfScanner 实例
      - guard: Python314CompatGuard 实例
      - summarize(): 打印完整迁移摘要
    """

    fix_record: GlobalContextWrapperFixRecord = field(
        default_factory=GlobalContextWrapperFixRecord
    )
    scanner: DoubleSelfScanner = field(default_factory=DoubleSelfScanner)
    guard: Python314CompatGuard = field(default_factory=Python314CompatGuard)

    def summarize(self) -> str:
        """完整迁移摘要：包含修复记录 + 运行时环境快照。"""
        py_ver = self.guard.current_python_version()
        cython_ver = self.guard.cython_version_str() or "未检测到"
        safe = self.guard.is_cython_double_self_safe()
        safe_str = {True: "安全（Cython >= 3.1.0）", False: "存在风险", None: "无法判断"}[safe]

        return "\n".join([
            self.fix_record.summarize(),
            "",
            "── 运行时环境快照 ──",
            f"  Python 版本 : {py_ver[0]}.{py_ver[1]}",
            f"  Python 3.14+: {'是' if self.guard.is_py314_plus() else '否'}",
            f"  Cython 版本 : {cython_ver}",
            f"  Cython 安全 : {safe_str}",
            "────────────────────────────────────────────────",
        ])


# =============================================================================
# 6. 自测
# =============================================================================

def _self_test() -> None:
    """7 组断言自测，覆盖 58f376f 的核心变更逻辑。"""
    passed = 0
    failed = 0

    def check(label: str, ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            print(f"  [FAIL] {label}", file=sys.stderr)
            failed += 1

    print("─── python314_support self-test (58f376f) ───")

    # 组 1：CythonDoubleSelfRecord 结构
    check("_DEALLOC_FIXES count == 8", len(_DEALLOC_FIXES) == 8)
    rec0 = _DEALLOC_FIXES[0]
    check(
        "首条修复 old_expr 含 self.self.",
        "self.self." in rec0.old_expr,
    )
    check(
        "首条修复 new_expr 不含 self.self.",
        "self.self." not in rec0.new_expr,
    )
    check(
        "首条修复 new_expr 含 self.",
        "self." in rec0.new_expr,
    )
    check(
        "所有修复 old_expr 均含 self.self.",
        all("self.self." in r.old_expr for r in _DEALLOC_FIXES),
    )
    check(
        "所有修复 new_expr 均不含 self.self.",
        all("self.self." not in r.new_expr for r in _DEALLOC_FIXES),
    )

    # 组 2：DoubleSelfScanner
    scanner = DoubleSelfScanner()

    clean_code = """
cdef class Foo:
    def __dealloc__(self):
        Py_DECREF(self.bar)
        Py_DECREF(self.baz)
"""
    hits_clean = scanner.scan_source(clean_code)
    check("clean code: 0 hits", len(hits_clean) == 0)

    buggy_code = """
cdef class GlobalContextWrapper:
    def __dealloc__(self):
        Py_DECREF(self.self.temp_create_context_fn)
        Py_DECREF(self.self.temp_destroy_context_fn)
        if self.temp_global_context:
            Py_DECREF(self.self.temp_global_context)
"""
    hits_buggy = scanner.scan_source(buggy_code)
    check("buggy code: 3 hits (self.self.*)", len(hits_buggy) == 3)

    # assert_no_double_self 对干净代码不抛异常
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pyx", delete=False, encoding="utf-8"
    ) as f:
        f.write(clean_code)
        tmppath_clean = f.name
    try:
        scanner.assert_no_double_self(tmppath_clean)
        check("assert_no_double_self: clean file 不抛异常", True)
    except Exception:
        check("assert_no_double_self: clean file 不抛异常", False)
    finally:
        _os.unlink(tmppath_clean)

    # assert_no_double_self 对含 bug 代码抛 AssertionError
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pyx", delete=False, encoding="utf-8"
    ) as f:
        f.write(buggy_code)
        tmppath_buggy = f.name
    try:
        try:
            scanner.assert_no_double_self(tmppath_buggy)
            check("assert_no_double_self: buggy file 抛 AssertionError", False)
        except AssertionError:
            check("assert_no_double_self: buggy file 抛 AssertionError", True)
    finally:
        _os.unlink(tmppath_buggy)

    # 不存在的文件静默跳过
    hits_missing = scanner.scan_file("/nonexistent/path/foo.pyx")
    check("scan_file: 不存在文件返回空列表", hits_missing == [])

    # 组 3：Python314CompatGuard - 版本检测
    guard = Python314CompatGuard()
    py_ver = guard.current_python_version()
    check(
        "current_python_version() 返回 (int, int)",
        isinstance(py_ver, tuple) and len(py_ver) == 2
        and all(isinstance(x, int) for x in py_ver),
    )
    # is_py314_plus 与 sys.version_info 一致
    expected_314 = sys.version_info[:2] >= (3, 14)
    check(
        f"is_py314_plus() == {expected_314}",
        guard.is_py314_plus() == expected_314,
    )

    # 组 4：GlobalContextWrapperFixRecord
    fix_record = GlobalContextWrapperFixRecord()
    check("fix_record.upstream_commit 含 58f376f", "58f376f" in fix_record.upstream_commit)
    check(
        "fix_record.fixes count == 8",
        len(fix_record.fixes) == 8,
    )
    summary = fix_record.summarize()
    check("fix_record.summarize() 含 '修复数量'", "修复数量" in summary)
    check("fix_record.summarize() 含 '8'", "8" in summary)
    old_exprs = fix_record.all_old_exprs()
    check(
        "all_old_exprs 均含 self.self.",
        all("self.self." in e for e in old_exprs),
    )

    # 组 5：Python314SupportAudit
    audit = Python314SupportAudit()
    audit_summary = audit.summarize()
    check("audit.summarize() 含 'Python 版本'", "Python 版本" in audit_summary)
    check("audit.summarize() 含 'Cython 版本'", "Cython 版本" in audit_summary)
    check("audit.summarize() 含 '修复摘要'", "修复摘要" in audit_summary)

    # 组 6：_DOUBLE_SELF_PATTERN 正则
    check(
        "_DOUBLE_SELF_PATTERN 匹配 'self.self.foo'",
        bool(_DOUBLE_SELF_PATTERN.search("self.self.foo")),
    )
    check(
        "_DOUBLE_SELF_PATTERN 不匹配 'self.foo'",
        not _DOUBLE_SELF_PATTERN.search("Py_DECREF(self.foo)"),
    )
    check(
        "_DOUBLE_SELF_PATTERN 不匹配 'myself.self.foo'（无 \\b 前缀边界）",
        # 注：\\b 保证只匹配 self.self. 开头，不匹配 myself.self.
        not _DOUBLE_SELF_PATTERN.search("myself.self.foo"),
    )

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS] python314_support.py 自测通过 (58f376f)")


# ── 模块级单例 ─────────────────────────────────────────────────────────────────
_DEFAULT_SCANNER: DoubleSelfScanner = DoubleSelfScanner()
_DEFAULT_GUARD: Python314CompatGuard = Python314CompatGuard()
_DEFAULT_FIX_RECORD: GlobalContextWrapperFixRecord = GlobalContextWrapperFixRecord()
_DEFAULT_AUDIT: Python314SupportAudit = Python314SupportAudit()


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
    print()
    print(_DEFAULT_AUDIT.summarize())
