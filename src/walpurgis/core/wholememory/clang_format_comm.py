"""
clang_format_comm.py — 5a9accc 迁移: run clang format (communicator.cpp)

migrate 5a9accc: run clang format

上游来源: cpp/src/wholememory/communicator.cpp
commit:   5a9accc8a85403e41e827b9aa3a3d85fec23d9f4
Author:   Alexandria Barghi <abarghi@nvidia.com>
Date:     Sun Dec 7 21:11:54 2025 -0800

上游变更本质 (1 file changed, 2 insertions(+), 4 deletions(-)):
  cpp/src/wholememory/communicator.cpp 两处 clang-format 格式化:

  1. get_host_name() — gethostname 错误分支从三行展开压缩为单行 inline:
     旧 (三行):
       if (gethostname(hostname, maxlen) != 0) {
         WHOLEMEMORY_FATAL("gethostname failed.");
       }
     新 (单行):
       if (gethostname(hostname, maxlen) != 0) { WHOLEMEMORY_FATAL("gethostname failed."); }

  2. get_boot_id() — copy_len 赋值加对齐空格:
     旧: size_t copy_len = std::min(strlen(p), remaining);
     新: size_t copy_len  = std::min(strlen(p), remaining);
     (双空格对齐上方 size_t remaining = len - offset - 1;)

Python 侧迁移位置:
  上游 communicator.cpp 中 get_host_name() / get_boot_id() 已于
  migrate 3626464 (ipc_guard.py) 做过 Python 防护层封装:
    - HostnameGuard.get_delimited()  ← 对应 get_host_name()
    - BootIdBuilder.build()          ← 对应 get_boot_id()

  5a9accc 是纯格式化 commit，无语义变更，Python 层不存在等价代码风格差异。

  Walpurgis 鲁迅拿法 20% 改写:
  本模块将 ipc_guard.py 中两个函数的「格式化契约」显式文档化:
  1. HostnameFormatInvariant — 记录 get_host_name() 单行 inline 格式
     对应 Python 层的「单条件单行表达式」风格约定
  2. AlignedAssignmentDemo   — 记录 get_boot_id() copy_len 对齐格式
     对应 Python 层的「对齐赋值」代码风格约定
  3. StyleAudit.check()      — 扫描 ipc_guard.py 验证风格契约是否被维持
  4. WALPURGIS_DEBUG 断点:
     断点1: StyleAudit.check() 入口，打印被扫描文件路径
     断点2: 各契约检查点结果打印

鲁迅: 「横眉冷对千夫指，俯首甘为孺子牛。」
应用: 对上游「只是格式化」的 commit 也不敷衍了事——
      格式即契约，Walpurgis 将上游隐式的 clang-format 风格规则
      显式编码为可验证的 Python 层风格约定。

作者: dylanyunlon<dogechat@163.com>
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple

# ──────────────────────────────────────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────────────────────────────────────

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点打印: WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _DEBUG:
        print(
            f"[WALPURGIS wholememory/clang_format_comm|{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 上游 5a9accc 格式化规则文档化
#
# clang-format 的核心决策 (LLVM style, single-line threshold):
#   1. 短单语句 if-body 若整行 <= ColumnLimit(100) 则 inline → 单行
#   2. 相邻同类型变量声明的赋值符号可对齐 (AlignConsecutiveAssignments)
# ──────────────────────────────────────────────────────────────────────────────

# 5a9accc diff hunk 1: get_host_name() gethostname 错误分支
_UPSTREAM_BEFORE_HUNK1 = (
    'if (gethostname(hostname, maxlen) != 0) {\n'
    '    WHOLEMEMORY_FATAL("gethostname failed.");\n'
    '  }'
)
_UPSTREAM_AFTER_HUNK1 = (
    'if (gethostname(hostname, maxlen) != 0) { WHOLEMEMORY_FATAL("gethostname failed."); }'
)

# 5a9accc diff hunk 2: get_boot_id() copy_len 对齐赋值
_UPSTREAM_BEFORE_HUNK2 = 'size_t copy_len = std::min(strlen(p), remaining);'
_UPSTREAM_AFTER_HUNK2  = 'size_t copy_len  = std::min(strlen(p), remaining);'


class _FormatRule(NamedTuple):
    """描述一条 clang-format 规则的结构化记录。"""
    rule_id: str      # e.g. "5a9accc-hunk1"
    cpp_before: str   # 格式化前的 C++ 代码片段
    cpp_after: str    # 格式化后的 C++ 代码片段
    py_analog: str    # Python 层对应的风格约定描述
    clang_option: str # 对应的 clang-format 配置选项


# 两条规则的完整记录
CLANG_FORMAT_RULES: List[_FormatRule] = [
    _FormatRule(
        rule_id="5a9accc-hunk1",
        cpp_before=_UPSTREAM_BEFORE_HUNK1,
        cpp_after=_UPSTREAM_AFTER_HUNK1,
        py_analog=(
            "单条件单行表达式: "
            "`if condition: raise RuntimeError(...)` 而非三行展开，"
            "当 if-body 只有一个简短语句且总行长 <= 100 字符时使用。"
            "对应 HostnameGuard.get_hostname() 中 gethostname 失败分支。"
        ),
        clang_option="AllowShortBlocksOnASingleLine: Always (LLVM style)",
    ),
    _FormatRule(
        rule_id="5a9accc-hunk2",
        cpp_before=_UPSTREAM_BEFORE_HUNK2,
        cpp_after=_UPSTREAM_AFTER_HUNK2,
        py_analog=(
            "对齐赋值: 同一语义块内相邻变量赋值时，"
            "用空格对齐 `=` 号可提升可读性。"
            "对应 BootIdBuilder.build() 中 remaining / copy_len 赋值对。"
            "Python 中通常用 `remaining = ...` / `copy_len  = ...` 对齐。"
        ),
        clang_option="AlignConsecutiveAssignments: Consecutive",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# HostnameFormatInvariant
#
# 将 5a9accc hunk1 的 Python 层等价约定固定为可检查的不变量。
#
# 上游 C++:
#   旧: if (gethostname(hostname, maxlen) != 0) {
#         WHOLEMEMORY_FATAL("gethostname failed.");
#       }
#   新: if (gethostname(hostname, maxlen) != 0) { WHOLEMEMORY_FATAL("gethostname failed."); }
#
# Python 等价: HostnameGuard.get_hostname() 中失败分支
#   约定: OSError 捕获块 body 应能在 ≤ 100 字符内表达
#         (若将 raise RuntimeError(...) 压缩为单行 inline)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HostnameFormatInvariant:
    """
    5a9accc hunk1 的 Python 层格式契约:
    单条件错误分支应尽量 inline（<=100 字符）。

    Invariant: get_hostname() OSError 分支逻辑可表达为单行 raise。
    """

    max_inline_len: int = 100  # clang-format ColumnLimit

    def check_inline_feasible(self, condition_str: str, body_str: str) -> bool:
        """
        检查 if condition: body 能否 inline（总长 <= max_inline_len）。

        参数
        ----
        condition_str : str  e.g. "gethostname(hostname, maxlen) != 0"
        body_str      : str  e.g. 'WHOLEMEMORY_FATAL("gethostname failed.")'

        返回
        ----
        bool: True = 可 inline，False = 需展开

        断点2: 打印 inline 可行性判断
        """
        candidate = f"if {condition_str}: {body_str}"
        feasible = len(candidate) <= self.max_inline_len

        # ── 断点2 ─────────────────────────────────────────────────────────
        _dbg(
            "hostname_fmt",
            f"check_inline_feasible: len={len(candidate)} "
            f"max={self.max_inline_len} feasible={feasible}\n"
            f"  candidate={candidate!r}",
        )
        return feasible

    def assert_ipc_guard_style(self) -> None:
        """
        验证 ipc_guard.py 的 HostnameGuard.get_hostname() 遵循 inline 约定。

        直接检查源码中 OSError 分支是否为单行 raise（即 inline 风格）。
        这是 5a9accc hunk1 在 Python 层的风格契约验证。

        断点2: 打印检查结果
        """
        try:
            from . import ipc_guard
            src = inspect.getsource(ipc_guard.HostnameGuard.get_hostname)
        except Exception as exc:
            _dbg("hostname_fmt", f"assert_ipc_guard_style: 无法获取源码 ({exc})，跳过")
            return

        # 上游 hunk1 的 Python 等价: OSError 分支中 raise 与 except 在同一语义组
        # 检查: except OSError 块内没有独立三行 if/body/end 结构
        has_three_line_pattern = bool(
            re.search(r"except\s+OSError.*:\s*\n\s+if\s+", src, re.DOTALL)
        )

        # ── 断点2 ─────────────────────────────────────────────────────────
        _dbg(
            "hostname_fmt",
            f"assert_ipc_guard_style: has_three_line_if_in_except={has_three_line_pattern}",
        )

        if has_three_line_pattern:
            import warnings
            warnings.warn(
                "[clang_format_comm] ipc_guard.HostnameGuard.get_hostname() "
                "中 OSError 分支存在三行展开 if，"
                "建议按 5a9accc hunk1 风格改为单行 inline。",
                UserWarning,
                stacklevel=2,
            )
        else:
            _dbg("hostname_fmt", "✓ ipc_guard OSError 分支风格符合 5a9accc hunk1 约定")


# ──────────────────────────────────────────────────────────────────────────────
# AlignedAssignmentDemo
#
# 将 5a9accc hunk2 的 Python 层等价约定固定为可检查的不变量。
#
# 上游 C++:
#   旧: size_t copy_len = std::min(strlen(p), remaining);
#   新: size_t copy_len  = std::min(strlen(p), remaining);
#   (双空格: AlignConsecutiveAssignments 使 copy_len 与上行 remaining 的 = 对齐)
#
# Python 等价: BootIdBuilder.build() 中
#   remaining = max_chars         (或 max_chars - offset 之类)
#   copy_len  = raw[:remaining]   ← 双空格对齐
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignedAssignmentDemo:
    """
    5a9accc hunk2 的 Python 层格式契约:
    同语义块内相邻赋值用空格对齐 `=`。

    演示: 与 BootIdBuilder.build() 的 remaining / copy_len 对比。
    """

    def demo_aligned_pair(self) -> str:
        """
        返回对齐赋值的 Python 示范代码（对应 5a9accc hunk2）。

        上游 C++ (hunk2 after):
            size_t remaining = len - offset - 1;
            size_t copy_len  = std::min(strlen(p), remaining);  ← 对齐

        Python 等价 (Walpurgis BootIdBuilder.build() 风格):
            max_chars = self.maxlen - 1
            copy_len  = raw[:max_chars]                          ← 对齐
        """
        aligned_code = (
            "# 5a9accc hunk2: AlignConsecutiveAssignments 风格\n"
            "# C++ 原文 (after format):\n"
            "#   size_t remaining = len - offset - 1;\n"
            "#   size_t copy_len  = std::min(strlen(p), remaining);\n"
            "#\n"
            "# Python 等价 (BootIdBuilder.build):\n"
            "max_chars = self.maxlen - 1          # remaining\n"
            "copy_len  = len(raw[:max_chars])     # copy_len (对齐 =)\n"
        )
        _dbg("aligned_assign", f"demo_aligned_pair:\n{aligned_code}")
        return aligned_code

    def check_alignment(self, line_a: str, line_b: str) -> bool:
        """
        检查两行赋值语句的 `=` 是否对齐（列位置相同）。

        参数
        ----
        line_a : str  e.g. "max_chars = self.maxlen - 1"
        line_b : str  e.g. "copy_len  = raw[:max_chars]"

        返回
        ----
        bool: True = `=` 在同一列，False = 未对齐

        断点2: 打印对齐检查结果
        """
        col_a = line_a.index("=") if "=" in line_a else -1
        col_b = line_b.index("=") if "=" in line_b else -1
        aligned = (col_a >= 0 and col_b >= 0 and col_a == col_b)

        # ── 断点2 ─────────────────────────────────────────────────────────
        _dbg(
            "aligned_assign",
            f"check_alignment: col_a={col_a} col_b={col_b} aligned={aligned}\n"
            f"  line_a={line_a!r}\n"
            f"  line_b={line_b!r}",
        )
        return aligned


# ──────────────────────────────────────────────────────────────────────────────
# StyleAudit
#
# 主入口: 扫描 ipc_guard.py，验证 5a9accc 的两条格式化规则
# 是否在 Python 层被维持。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StyleAudit:
    """
    5a9accc 格式化规则的 Python 层审计器。

    使用:
        audit = StyleAudit()
        results = audit.check()
        for r in results:
            print(r)

    断点1: check() 入口打印被扫描文件路径
    断点2: 各规则检查结果
    """

    def _locate_ipc_guard(self) -> Path:
        """定位 ipc_guard.py 的文件系统路径。"""
        here = Path(__file__).parent
        candidate = here / "ipc_guard.py"
        if candidate.exists():
            return candidate
        # fallback: 通过 importlib 定位
        try:
            import importlib.util
            spec = importlib.util.find_spec(".ipc_guard", package=__package__)
            if spec and spec.origin:
                return Path(spec.origin)
        except Exception:
            pass
        return candidate  # 可能不存在，调用方处理

    def check(self) -> List[str]:
        """
        执行风格审计，返回审计报告行列表。

        断点1: 打印被扫描文件路径
        断点2: 各规则检查结果
        """
        report: List[str] = []
        ipc_guard_path = self._locate_ipc_guard()

        # ── 断点1 ─────────────────────────────────────────────────────────
        _dbg("audit", f"check(): 扫描文件 {ipc_guard_path}")

        if not ipc_guard_path.exists():
            msg = f"[StyleAudit] ipc_guard.py 未找到 ({ipc_guard_path})，跳过审计"
            _dbg("audit", msg)
            report.append(msg)
            return report

        src = ipc_guard_path.read_text(encoding="utf-8")
        report.append(f"[StyleAudit] 审计文件: {ipc_guard_path}")
        report.append(f"[StyleAudit] 对应上游 commit: 5a9accc (run clang format)")
        report.append(f"[StyleAudit] 规则数: {len(CLANG_FORMAT_RULES)}")

        # ── 规则 1: hunk1 — HostnameGuard inline 风格 ─────────────────────
        rule1 = CLANG_FORMAT_RULES[0]
        _dbg("audit", f"检查规则 {rule1.rule_id}: {rule1.clang_option}")

        # Python 等价检查: except OSError 块内不应有独立的三行 if 展开
        has_three_line_in_except = bool(
            re.search(r"except\s+OSError[^:]*:\s*\n\s+if\s+", src, re.DOTALL)
        )
        r1_status = "✓ PASS" if not has_three_line_in_except else "⚠ WARN"
        r1_msg = (
            f"[StyleAudit] {rule1.rule_id} ({rule1.clang_option}): {r1_status}\n"
            f"  Python 约定: {rule1.py_analog}\n"
            f"  检查: OSError 分支无三行展开 if → {r1_status}"
        )
        report.append(r1_msg)
        _dbg("audit", f"规则 {rule1.rule_id}: {r1_status}")

        # ── 规则 2: hunk2 — BootIdBuilder 对齐赋值 ────────────────────────
        rule2 = CLANG_FORMAT_RULES[1]
        _dbg("audit", f"检查规则 {rule2.rule_id}: {rule2.clang_option}")

        # Python 等价检查: BootIdBuilder.build() 中是否存在 copy_len 与 max_chars 对齐
        # 搜索模式: 连续两行赋值其中一行含 copy_len 或 remaining
        aligned_pattern = re.search(
            r"(max_chars\s*=\s*.+)\n\s*(copy_len\s+=\s*.+)",
            src,
        )
        if aligned_pattern:
            line_a = aligned_pattern.group(1).strip()
            line_b = aligned_pattern.group(2).strip()
            demo = AlignedAssignmentDemo()
            is_aligned = demo.check_alignment(line_a, line_b)
            r2_status = "✓ PASS" if is_aligned else "⚠ WARN"
        else:
            r2_status = "～ N/A"  # 源码已重构，无直接对应行
            line_a = line_b = "(未找到直接对应赋值对)"

        r2_msg = (
            f"[StyleAudit] {rule2.rule_id} ({rule2.clang_option}): {r2_status}\n"
            f"  Python 约定: {rule2.py_analog}\n"
            f"  检查: copy_len 与 max_chars 对齐赋值 → {r2_status}"
        )
        report.append(r2_msg)
        _dbg("audit", f"规则 {rule2.rule_id}: {r2_status}")

        # ── 汇总 ──────────────────────────────────────────────────────────
        pass_count = sum(1 for r in report if "✓ PASS" in r)
        warn_count = sum(1 for r in report if "⚠ WARN" in r)
        report.append(
            f"[StyleAudit] 汇总: {pass_count} PASS / {warn_count} WARN / "
            f"{len(CLANG_FORMAT_RULES) - pass_count - warn_count} N/A"
        )
        _dbg("audit", f"审计完成: {pass_count} PASS, {warn_count} WARN")

        return report

    def print_rules(self) -> None:
        """打印所有记录的 5a9accc clang-format 规则（调试/文档用）。"""
        print(f"[clang_format_comm] 5a9accc 格式化规则 ({len(CLANG_FORMAT_RULES)} 条):")
        for rule in CLANG_FORMAT_RULES:
            print(f"\n  [{rule.rule_id}] {rule.clang_option}")
            print(f"  C++ before: {rule.cpp_before[:80]!r}{'...' if len(rule.cpp_before)>80 else ''}")
            print(f"  C++ after : {rule.cpp_after[:80]!r}{'...' if len(rule.cpp_after)>80 else ''}")
            print(f"  Python    : {rule.py_analog}")


# ──────────────────────────────────────────────────────────────────────────────
# 便捷 API
# ──────────────────────────────────────────────────────────────────────────────

def run_style_audit(verbose: bool = False) -> List[str]:
    """
    执行 5a9accc 风格审计并返回报告。

    参数
    ----
    verbose : bool, default False
        True 时同时打印到 stdout。

    用法:
        from walpurgis.core.wholememory.clang_format_comm import run_style_audit
        report = run_style_audit(verbose=True)
    """
    audit = StyleAudit()
    report = audit.check()
    if verbose:
        for line in report:
            print(line)
    return report


__all__ = [
    "CLANG_FORMAT_RULES",
    "HostnameFormatInvariant",
    "AlignedAssignmentDemo",
    "StyleAudit",
    "run_style_audit",
]


# ──────────────────────────────────────────────────────────────────────────────
# 自测 (WALPURGIS_DEBUG=1 时运行)
# ──────────────────────────────────────────────────────────────────────────────

if _DEBUG:
    _dbg("selftest", "=== clang_format_comm 自测开始 ===")

    # 断点1 + 断点2: StyleAudit.check()
    _audit = StyleAudit()
    _report = _audit.check()
    for _line in _report:
        _dbg("selftest", _line)

    # HostnameFormatInvariant
    _hfi = HostnameFormatInvariant()
    _hfi.check_inline_feasible(
        "gethostname(hostname, maxlen) != 0",
        'WHOLEMEMORY_FATAL("gethostname failed.")',
    )
    _hfi.assert_ipc_guard_style()

    # AlignedAssignmentDemo
    _aad = AlignedAssignmentDemo()
    _aad.demo_aligned_pair()
    _aad.check_alignment(
        "max_chars = self.maxlen - 1",
        "copy_len  = raw[:max_chars]",
    )

    _dbg("selftest", "=== clang_format_comm 自测完成 ===")
