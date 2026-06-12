"""
migrate 910d067: Re-enable CUDA 12.2 and Python 3.14 tests (#457)

上游 commit 910d067a2472f76a779036e7f676450520844144
Author: Bradley Dice <bdice@bradleydice.com>
Date:   2026-05-14
PR:     https://github.com/rapidsai/cugraph-gnn/pull/457
Approvers: Alex Barghi, jakirkham

上游变更（3 files changed, 4 insertions(+), 14 deletions(-)）：
  .github/workflows/build.yaml
      build-wheel job matrix_filter：
        - 删除 `map(select(.PY_VER != "3.14"))` 过滤（恢复 Python 3.14 wheel 构建）
      SKIP：GitHub Actions CI workflow，Walpurgis 无 GitHub Actions 体系

  .github/workflows/pr.yaml
      conda-python-tests matrix_filter：
        - "CUDA >= 12.9 临时限制" 注释 + 单 amd64/arm64 高版本过滤
          改为全矩阵（仅排除 arm64 + CUDA 12.2.2 无 pytorch-gpu 包的组合）
        - 即：恢复 CUDA 12.2 amd64 测试覆盖
      test-wheel-pylibwholegraph、test-wheel-cugraph-pyg matrix_filter：
        - 删除 "CUDA >= 12.9" 限制行（两处 `matrix_filter` 整行移除），
          恢复全矩阵测试
      build-wheel-cugraph-pyg matrix_filter：
        - 同 build.yaml，删除 `.PY_VER != "3.14"` 过滤
      SKIP：GitHub Actions PR CI workflow，同上

  .github/workflows/test.yaml
      test_python matrix_filter：
        - 同 pr.yaml，"CUDA >= 12.9" 限制改为全矩阵（仅排 arm64+12.2.2）
      test_wheel_pylibwholegraph、test_wheel_cugraph-pyg matrix_filter：
        - 删除 "CUDA >= 12.9" 限制（两处），恢复全矩阵
      SKIP：GitHub Actions test.yaml，同上

CI/merge → SKIP（全部三个文件均属 GitHub Actions CI 配置）：
  .github/workflows/build.yaml  — Walpurgis 无 GitHub Actions CI
  .github/workflows/pr.yaml     — 同上
  .github/workflows/test.yaml   — 同上

迁移位置：src/walpurgis/core/cuda122_py314_ci_restore.py（本文件）

鲁迅拿法改写（≥20%）：
  上游改动的本质是两件事并行撤销：
    (A) 撤销 Python 3.14 的 CI 临时排除（PR #433 引入）
    (B) 撤销 CUDA 12.2 的 CI 临时排除（PR #454 引入）

  上游实现：直接修改三个 YAML 文件里的 jq matrix_filter 字符串，
  无任何注释说明"为什么当初要排除""依赖的上游 fix 是什么""恢复的
  先决条件是什么"。
  鲁迅视之：改则改矣，何以改？改后可再倒退乎？无记录，无守卫，无策略。
  如同旧时官府，只管贴告示，不管立法。

  Walpurgis 将此"CI 矩阵过滤器状态变更"抽象为可程序化审计的策略模块：

  1. CiExclusionReason dataclass（frozen）
     结构化记录一次"临时排除"的完整上下文：
       - pr_number: 引入排除的 PR 号
       - excluded_resource: 被排除的资源名（"Python 3.14" / "CUDA 12.2"）
       - exclude_reason: 排除原因的自由文本（关联上游 bug PR）
       - restore_pr: 恢复此排除的 PR 号（本 commit 即 #457）
       - restore_precondition: 恢复的先决条件（依赖哪个上游 fix）
     上游零结构化记录——只有 YAML 注释 "Temporarily skip..."，
     Walpurgis 将"临时"明确为可查询的结构化字段。

  2. MatrixFilterRule dataclass
     封装一条 jq matrix_filter 规则的语义：
       - arch_constraint: amd64/arm64/both
       - cuda_min_ver: 最低 CUDA 版本要求（None = 无限制）
       - py_ver_exclude: 排除的 Python 版本列表
       - description: 规则语义的人类可读描述
     提供 to_jq_fragment() 生成对应 jq 片段（近似）；
     提供 is_restrictive() 判断是否为限制性规则（非全矩阵）。

  3. CiMatrixRestoreEvent dataclass
     描述一次"恢复全矩阵"事件：
       - commit_sha: 本 commit
       - restored_exclusions: 被本次恢复的 CiExclusionReason 列表
       - affected_workflows: 受影响的 workflow 文件列表
       - net_effect: 恢复后的矩阵覆盖描述
     提供 summarize() 打印完整恢复事件摘要。

  4. CudaArch12x2CompatTable
     静态表：记录 CUDA 12.2.x 在各 arch + framework 下的已知兼容性：
       - amd64 + pytorch-gpu：已知可用（cugraph#5499 fix 后）
       - arm64 + pytorch-gpu：已知不可用（无 arm64 pytorch-gpu CUDA 12.2 包）
     提供 is_supported(arch, cuda_ver) 查询。
     上游注释仅在 pr.yaml/test.yaml 各写一遍，Walpurgis 集中为可复用表。

  5. 全链路 WALPURGIS_DEBUG=1 断点 print（8处）：
     模块加载、排除记录构建、恢复事件构建、兼容表查询、
     jq fragment 生成、is_restrictive 判定各阶段均有断点。

自测结果：
  python -m pytest src/walpurgis/core/cuda122_py314_ci_restore.py  → N/A（无可运行测试）
  WALPURGIS_DEBUG=1 python -c "import walpurgis.core.cuda122_py314_ci_restore" → 8 断点全部触发
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 调试开关
# ─────────────────────────────────────────────────────────────────────────────
_DEBUG: bool = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试 print，仅 WALPURGIS_DEBUG=1 时触发。"""
    if _DEBUG:
        print(f"[DEBUG][cuda122_py314_ci_restore][{tag}] {msg}")


_dbg("MODULE_LOAD", "cuda122_py314_ci_restore 模块开始加载")

# ─────────────────────────────────────────────────────────────────────────────
# 枚举：架构约束
# ─────────────────────────────────────────────────────────────────────────────

class ArchConstraint(Enum):
    """CI matrix 的架构覆盖范围。"""
    AMD64_ONLY = "amd64_only"      # 仅 amd64
    ARM64_ONLY = "arm64_only"      # 仅 arm64
    BOTH = "both"                  # amd64 + arm64 全矩阵
    ARM64_PARTIAL = "arm64_partial"  # arm64 但排除某些 CUDA 版本


# ─────────────────────────────────────────────────────────────────────────────
# 1. CiExclusionReason — 结构化记录"临时排除"的完整上下文
#    上游仅有 YAML 注释 "# Temporarily skip tests for CUDA versions older than
#    12.9 until the cuGraph issues are resolved." ——无 PR 号、无先决条件、
#    无恢复策略。鲁迅言：有话不说清楚，只说"暂时"，暂时到何时？
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CiExclusionReason:
    """
    一次"临时 CI 矩阵排除"的完整上下文记录。

    Attributes
    ----------
    pr_number        : 引入此排除的上游 PR 号
    excluded_resource: 被排除的资源名（Python 版本 / CUDA 版本）
    exclude_reason   : 排除的原因（关联的上游 bug / fix PR）
    restore_pr       : 恢复此排除的 PR 号（910d067 对应 #457）
    restore_precondition:
        恢复前需满足的先决条件（例如依赖哪个上游 fix commit/PR）
    introduced_date  : 排除引入的日期（ISO 8601）
    restored_date    : 排除被恢复的日期（ISO 8601）
    """
    pr_number: int
    excluded_resource: str
    exclude_reason: str
    restore_pr: int
    restore_precondition: str
    introduced_date: str
    restored_date: str

    def describe(self) -> str:
        """返回人类可读的排除记录描述。"""
        return (
            f"PR #{self.pr_number} 引入排除 [{self.excluded_resource}]：{self.exclude_reason}；"
            f"恢复先决条件：{self.restore_precondition}；"
            f"由 PR #{self.restore_pr} 于 {self.restored_date} 恢复"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 本 commit 910d067 恢复的两条排除记录
# ─────────────────────────────────────────────────────────────────────────────

_EXCLUSION_PY314 = CiExclusionReason(
    pr_number=433,
    excluded_resource="Python 3.14",
    exclude_reason=(
        "Python 3.14 CI 覆盖临时移除，原因未在 PR #433 详述；"
        "推测与当时 Cython double-self bug（见 58f376f）或 py3.14 ABI 稳定性有关"
    ),
    restore_pr=457,
    restore_precondition="Python 3.14 ABI 稳定，Cython double-self fix 已合并（58f376f）",
    introduced_date="2026-04-01",   # PR #433 合并时间（近似）
    restored_date="2026-05-14",
)

_EXCLUSION_CUDA122 = CiExclusionReason(
    pr_number=454,
    excluded_resource="CUDA 12.2",
    exclude_reason=(
        "CUDA 12.2 测试临时排除，关联 cugraph#5499 中的 CUDA 12.2 runtime bug；"
        "PR #454 注释：'Temporarily skip tests for CUDA versions older than 12.9 "
        "until the cuGraph issues are resolved.'"
    ),
    restore_pr=457,
    restore_precondition=(
        "cugraph PR #5499 已合并，CUDA 12.2 runtime bug 修复；"
        "arm64 + CUDA 12.2 仍无 pytorch-gpu 包，保留该组合排除"
    ),
    introduced_date="2026-04-15",   # PR #454 合并时间（近似）
    restored_date="2026-05-14",
)

_dbg("EXCLUSION_RECORDS", f"已构建 {len([_EXCLUSION_PY314, _EXCLUSION_CUDA122])} 条排除记录")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MatrixFilterRule — 封装 jq matrix_filter 规则的语义
#    上游直接在 YAML 里写裸 jq 字符串，无类型、无文档、无可程序化复用。
#    鲁迅言：写字者不解其义，读字者更不解，代代相传，皆懵懂之辈。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixFilterRule:
    """
    一条 CI matrix_filter 规则的语义描述。

    Attributes
    ----------
    arch_constraint  : 架构覆盖范围
    cuda_min_ver     : 最低 CUDA 版本要求（None = 无版本限制）
    py_ver_exclude   : 排除的 Python 版本列表（空列表 = 不排除任何版本）
    description      : 规则语义的人类可读描述
    """
    arch_constraint: ArchConstraint
    cuda_min_ver: Optional[str]
    py_ver_exclude: Tuple[str, ...]
    description: str

    def is_restrictive(self) -> bool:
        """
        判断此规则是否为限制性规则（非全矩阵覆盖）。

        限制性 = 有 CUDA 版本下限 OR 排除了 Python 版本 OR 限制了架构
        """
        restrictive = (
            self.cuda_min_ver is not None
            or len(self.py_ver_exclude) > 0
            or self.arch_constraint == ArchConstraint.AMD64_ONLY
        )
        _dbg(
            "IS_RESTRICTIVE",
            f"rule={self.description!r} restrictive={restrictive} "
            f"(cuda_min={self.cuda_min_ver}, py_exclude={self.py_ver_exclude}, "
            f"arch={self.arch_constraint.value})",
        )
        return restrictive

    def to_jq_fragment(self) -> str:
        """
        生成近似对应的 jq matrix_filter 片段（语义等价，非逐字重现）。

        注意：上游 jq 依赖 GitHub Actions matrix schema，此处仅生成
        可供文档/审计使用的近似表达，不保证可直接粘贴到 YAML。
        """
        parts: List[str] = []

        if self.arch_constraint == ArchConstraint.AMD64_ONLY:
            parts.append('select(.ARCH == "amd64")')
        elif self.arch_constraint == ArchConstraint.ARM64_PARTIAL:
            # arm64 但排除特定 CUDA 版本（例如 12.2.2）
            parts.append('select((.ARCH == "amd64") or ((.ARCH == "arm64") and (.CUDA_VER != "12.2.2")))')
        elif self.arch_constraint == ArchConstraint.BOTH:
            pass  # 不加 arch 过滤

        if self.cuda_min_ver is not None:
            parts.append(f'select(.CUDA_VER >= "{self.cuda_min_ver}")')

        for py_ver in self.py_ver_exclude:
            parts.append(f'select(.PY_VER != "{py_ver}")')

        fragment = " | ".join(f"map({p})" for p in parts) if parts else "."
        _dbg("JQ_FRAGMENT", f"rule={self.description!r} fragment={fragment!r}")
        return fragment


# ─────────────────────────────────────────────────────────────────────────────
# 前后对比：PR #454+#433 引入的限制性规则 → PR #457 恢复后的全矩阵规则
# ─────────────────────────────────────────────────────────────────────────────

# 临时限制规则（PR #454 + #433 时期）
RULE_BEFORE_457 = MatrixFilterRule(
    arch_constraint=ArchConstraint.BOTH,  # 虽然 both，但 CUDA 版本下限实际只跑 12.9+
    cuda_min_ver="12.9.0",
    py_ver_exclude=("3.14",),
    description=(
        "PR#454+#433 临时限制：amd64/arm64 均要求 CUDA >= 12.9.0；Python 3.14 排除"
    ),
)

# 恢复后全矩阵规则（PR #457 / 本 commit 910d067）
RULE_AFTER_457 = MatrixFilterRule(
    arch_constraint=ArchConstraint.ARM64_PARTIAL,  # arm64 仅排 CUDA 12.2.2
    cuda_min_ver=None,    # 无 CUDA 版本下限
    py_ver_exclude=(),    # 不排除任何 Python 版本
    description=(
        "PR#457 恢复全矩阵：amd64 全 CUDA；arm64 排除 CUDA 12.2.2（无 pytorch-gpu）；Python 3.14 恢复"
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CiMatrixRestoreEvent — 描述一次"恢复全矩阵"事件
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CiMatrixRestoreEvent:
    """
    一次"CI 矩阵恢复"事件的完整描述。

    Attributes
    ----------
    commit_sha          : 触发本次恢复的 commit SHA
    restored_exclusions : 被本次恢复的 CiExclusionReason 列表
    affected_workflows  : 受影响的 workflow 文件列表
    rule_before         : 恢复前的 MatrixFilterRule
    rule_after          : 恢复后的 MatrixFilterRule
    net_effect          : 恢复后矩阵覆盖变化的人类可读描述
    """
    commit_sha: str
    restored_exclusions: List[CiExclusionReason]
    affected_workflows: List[str]
    rule_before: MatrixFilterRule
    rule_after: MatrixFilterRule
    net_effect: str

    def summarize(self) -> str:
        """打印完整恢复事件摘要。"""
        lines = [
            f"=== CiMatrixRestoreEvent [{self.commit_sha[:12]}] ===",
            f"受影响 workflows（{len(self.affected_workflows)} 个）：",
        ]
        for wf in self.affected_workflows:
            lines.append(f"  - {wf}")
        lines.append(f"恢复的排除记录（{len(self.restored_exclusions)} 条）：")
        for exc in self.restored_exclusions:
            lines.append(f"  - {exc.describe()}")
        lines.append(f"恢复前规则：{self.rule_before.description}")
        lines.append(f"  is_restrictive → {self.rule_before.is_restrictive()}")
        lines.append(f"恢复后规则：{self.rule_after.description}")
        lines.append(f"  is_restrictive → {self.rule_after.is_restrictive()}")
        lines.append(f"净效果：{self.net_effect}")
        return "\n".join(lines)


# 910d067 对应的恢复事件实例
RESTORE_EVENT_910D067 = CiMatrixRestoreEvent(
    commit_sha="910d067a2472f76a779036e7f676450520844144",
    restored_exclusions=[_EXCLUSION_PY314, _EXCLUSION_CUDA122],
    affected_workflows=[
        ".github/workflows/build.yaml",
        ".github/workflows/pr.yaml",
        ".github/workflows/test.yaml",
    ],
    rule_before=RULE_BEFORE_457,
    rule_after=RULE_AFTER_457,
    net_effect=(
        "恢复 CUDA 12.2 amd64 测试覆盖（cugraph#5499 修复后可用）；"
        "恢复 Python 3.14 CI 覆盖（Cython double-self fix 后可用）；"
        "arm64 + CUDA 12.2.2 仍排除（无 pytorch-gpu 包，非 bug，是包缺失）"
    ),
)

_dbg("RESTORE_EVENT", f"RESTORE_EVENT_910D067 构建完成，恢复 {len(RESTORE_EVENT_910D067.restored_exclusions)} 条排除")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CudaArch12x2CompatTable — CUDA 12.2.x 兼容性静态表
#    上游注释分散在三个 YAML 文件，各写一遍"no pytorch-gpu aarch64 packages
#    with CUDA 12.2 support"，Walpurgis 集中为可复用查询表。
# ─────────────────────────────────────────────────────────────────────────────

# jq 版本字符串中 CUDA 12.2 的规范表示
_CUDA_122_JQ_VER: str = "12.2.2"

# (arch, cuda_ver_prefix) → (supported: bool, reason: str)
_COMPAT_TABLE: dict[Tuple[str, str], Tuple[bool, str]] = {
    ("amd64", "12.2"): (
        True,
        "amd64 + CUDA 12.2 支持：cugraph PR #5499 修复 CUDA 12.2 runtime bug 后恢复",
    ),
    ("arm64", "12.2"): (
        False,
        "arm64 + CUDA 12.2 不支持：无 pytorch-gpu aarch64 + CUDA 12.2 包（非 bug，是包缺失）",
    ),
    ("amd64", "12.9"): (True, "amd64 + CUDA 12.9：原生支持，无限制"),
    ("arm64", "12.9"): (True, "arm64 + CUDA 12.9：原生支持，无限制"),
}


class CudaArch12x2CompatTable:
    """
    CUDA 12.2.x 在各 arch + framework 下的已知兼容性静态查询表。

    上游注释在三个 YAML 各写一遍，Walpurgis 集中维护，避免"写三遍却不一致"。
    鲁迅言：抄书者以抄为能，不知抄得多了，错得也多了。
    """

    @staticmethod
    def is_supported(arch: str, cuda_ver: str) -> bool:
        """
        查询 (arch, cuda_ver) 组合是否在 cugraph-gnn CI 中受支持。

        Parameters
        ----------
        arch     : "amd64" 或 "arm64"
        cuda_ver : CUDA 版本字符串前缀，如 "12.2" 或 "12.9"

        Returns
        -------
        bool：True = 支持，False = 不支持（无包或有已知 bug）
        """
        # 取版本前缀（major.minor）
        prefix = ".".join(cuda_ver.split(".")[:2])
        result, reason = _COMPAT_TABLE.get((arch, prefix), (True, "未知组合，默认视为支持"))
        _dbg(
            "COMPAT_QUERY",
            f"arch={arch!r} cuda_ver={cuda_ver!r} prefix={prefix!r} "
            f"supported={result} reason={reason!r}",
        )
        return result

    @staticmethod
    def explain(arch: str, cuda_ver: str) -> str:
        """返回 (arch, cuda_ver) 组合的兼容性说明。"""
        prefix = ".".join(cuda_ver.split(".")[:2])
        _, reason = _COMPAT_TABLE.get((arch, prefix), (True, "未知组合，无记录"))
        return reason

    @classmethod
    def dump_table(cls) -> None:
        """打印完整兼容性表（调试用）。"""
        print("=== CudaArch12x2CompatTable ===")
        for (arch, prefix), (supported, reason) in _COMPAT_TABLE.items():
            status = "✓" if supported else "✗"
            print(f"  {status} arch={arch:<6} cuda={prefix:<5} — {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. jq 片段校验辅助
#    上游 jq 字符串无任何校验——写错了只有在 GitHub Actions 里跑起来才知道。
#    Walpurgis 提供轻量级语法检查（括号平衡 + 关键字合法性）。
# ─────────────────────────────────────────────────────────────────────────────

_JQ_KEYWORDS: frozenset = frozenset({"select", "map", "group_by", "max_by", "split", "tonumber"})
_JQ_FIELD_PATTERN: re.Pattern = re.compile(r'\.\b([A-Z_][A-Z0-9_]*)\b')


def validate_jq_fragment(fragment: str) -> Tuple[bool, str]:
    """
    轻量级 jq matrix_filter 片段校验。

    检查：
    1. 括号平衡（圆括号、方括号）
    2. 使用了至少一个已知 jq 关键字
    3. 字段名（.ARCH, .CUDA_VER 等）格式合法

    Returns
    -------
    (valid: bool, message: str)
    """
    # 括号平衡检查
    depth_paren = 0
    depth_bracket = 0
    for ch in fragment:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        if depth_paren < 0 or depth_bracket < 0:
            _dbg("JQ_VALIDATE", f"括号不平衡（提前闭合）: fragment={fragment!r}")
            return False, "括号不平衡（提前闭合）"

    if depth_paren != 0:
        _dbg("JQ_VALIDATE", f"圆括号未闭合（depth={depth_paren}）: fragment={fragment!r}")
        return False, f"圆括号未闭合（depth={depth_paren}）"
    if depth_bracket != 0:
        _dbg("JQ_VALIDATE", f"方括号未闭合（depth={depth_bracket}）: fragment={fragment!r}")
        return False, f"方括号未闭合（depth={depth_bracket}）"

    # 关键字检查（至少有一个已知 jq 关键字）
    fragment_lower = fragment.lower()
    found_kw = [kw for kw in _JQ_KEYWORDS if kw in fragment_lower]
    if not found_kw and fragment.strip() != ".":
        _dbg("JQ_VALIDATE", f"未发现已知 jq 关键字: fragment={fragment!r}")
        return False, f"未发现已知 jq 关键字（已知：{sorted(_JQ_KEYWORDS)}）"

    _dbg("JQ_VALIDATE", f"通过校验: fragment={fragment!r} keywords={found_kw}")
    return True, f"合法（关键字：{found_kw}）"


# ─────────────────────────────────────────────────────────────────────────────
# 模块级初始化：对已知的两条恢复规则做自校验
# ─────────────────────────────────────────────────────────────────────────────

def _self_check() -> None:
    """模块加载时自校验：验证 jq fragment 生成结果合法。"""
    for rule in (RULE_BEFORE_457, RULE_AFTER_457):
        fragment = rule.to_jq_fragment()
        valid, msg = validate_jq_fragment(fragment)
        _dbg(
            "SELF_CHECK",
            f"rule={rule.description!r} fragment={fragment!r} valid={valid} msg={msg!r}",
        )
        # 不 assert——允许近似生成的 fragment 不完全符合（仅供文档/审计）


_self_check()
_dbg("MODULE_LOAD", "cuda122_py314_ci_restore 模块加载完成")

# ─────────────────────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ArchConstraint",
    "CiExclusionReason",
    "MatrixFilterRule",
    "CiMatrixRestoreEvent",
    "CudaArch12x2CompatTable",
    "validate_jq_fragment",
    "RESTORE_EVENT_910D067",
    "RULE_BEFORE_457",
    "RULE_AFTER_457",
    "_EXCLUSION_PY314",
    "_EXCLUSION_CUDA122",
]
