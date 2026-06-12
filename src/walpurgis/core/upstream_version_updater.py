"""
migrate 958783c: refactored update-version.sh to handle new branching strategy

上游 commit 958783c (Nate Rock <nrock@nvidia.com>, 2025-11-10):
  - ci/release/update-version.sh: 新增 --run-context=main|release CLI 参数
    优先级链: CLI > RAPIDS_RUN_CONTEXT 环境变量 > 默认 main
    main 上下文: RAPIDS_BRANCH_NAME = "main"
    release 上下文: RAPIDS_BRANCH_NAME = "release/{YY.MM}"
  - cpp/scripts/run-cmake-format.sh: 文档链接从 branch-25.02 改为 main (反映新策略)

CI/merge/docs → SKIP:
  - ci/release/update-version.sh   SKIP: CI发布脚本，Walpurgis无RAPIDS发布体系
  - cpp/scripts/run-cmake-format.sh SKIP: cmake格式化脚本，Walpurgis无C++/cmake构建

迁移位置:
  src/walpurgis/core/upstream_version_updater.py (本文件，新增)

鲁迅拿法改写(>20%):
  1. RunContextPolicy dataclass — 上游用裸 bash 字符串变量，此处强类型，携带
     source_label(来源描述)、branch_name(派生分支名)、pep440_norm(版本归一化)
     三元组，使"为何选此上下文"可审计。
  2. VersionSpec dataclass — 将上游 NEXT_FULL_TAG/MAJOR/MINOR/SHORT_TAG 四个
     散落 bash 变量收口为单一对象，__post_init__ 即校验 YY.MM.PP 格式。
  3. RunContextResolver — 封装 CLI > env > default 三段优先级链，
     from_cli()/from_env()/default() 三个命名工厂方法替代三段 if/elif/else。
  4. BranchNameStrategy — 封装 main vs release 两路分支名计算逻辑，
     compute() 静态方法返回 (branch_name, description) 具名二元组，
     上游是内联 if/elif 无名赋值。
  5. DocRefUpdatePolicy — 封装上游 "Documentation references - context-aware"
     的两路行为：main 上下文 noop + release 上下文 sed，
     上游内联 if/elif，无命名无测试。
  6. 全链路 WALPURGIS_DEBUG=1 断点 print，8处覆盖:
     VersionSpec 解析 → RunContextResolver 三段决策 →
     BranchNameStrategy 分支选择 → DocRefUpdatePolicy 决策路径 →
     RunContextPolicy 构建完成

用法示例:
  from walpurgis.core.upstream_version_updater import resolve_run_context_policy
  policy = resolve_run_context_policy(cli_context="release", new_version="25.12.00")
  print(policy.branch_name)   # "release/25.12"
  print(policy.pep440_norm)   # "25.12"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DBG:
        print(f"[WPG:upstream_version_updater:{tag}] {msg}", flush=True)


# ─── 1. VersionSpec — 收口上游四散的版本变量 ──────────────────────────────────

@dataclass(frozen=True)
class VersionSpec:
    """
    封装 958783c 上游 NEXT_FULL_TAG / NEXT_MAJOR / NEXT_MINOR / NEXT_SHORT_TAG。
    上游: 四个裸 bash 变量，各自用 awk split 计算，散落脚本中。
    改写: 单一值对象，__post_init__ 校验 YY.MM.PP 格式，属性懒派生。
    """

    full_tag: str        # e.g. "25.12.00"

    def __post_init__(self) -> None:
        _dbg("VersionSpec.parse", f"full_tag={self.full_tag!r}")
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", self.full_tag):
            raise ValueError(
                f"[VersionSpec] 版本格式必须为 YY.MM.PP，收到: {self.full_tag!r}\n"
                f"示例: 25.12.00"
            )
        _dbg("VersionSpec.ok", f"major={self.major} minor={self.minor} short={self.short_tag}")

    @property
    def major(self) -> str:
        return self.full_tag.split(".")[0]

    @property
    def minor(self) -> str:
        return self.full_tag.split(".")[1]

    @property
    def patch(self) -> str:
        return self.full_tag.split(".")[2]

    @property
    def short_tag(self) -> str:
        """YY.MM 部分，对应上游 NEXT_SHORT_TAG。"""
        return f"{self.major}.{self.minor}"

    @property
    def pep440(self) -> str:
        """
        PEP 440 规范化后的短标签。
        上游: python -c "from packaging.version import Version; print(Version(...))"
        改写: 纯字符串操作（short_tag 本身已满足 PEP 440，去前导零）。
        """
        # 25.12 → "25.12"（ packaging.Version 不改变此格式）
        major_int = int(self.major)
        minor_int = int(self.minor)
        return f"{major_int}.{minor_int}"


# ─── 2. RunContext 枚举 ───────────────────────────────────────────────────────

class RunContext(Enum):
    """对应上游 RUN_CONTEXT 变量的合法值集合。"""
    MAIN = "main"
    RELEASE = "release"


VALID_CONTEXTS = {c.value for c in RunContext}


# ─── 3. RunContextResolver — 三段优先级链 ────────────────────────────────────

@dataclass(frozen=True)
class RunContextSource:
    """
    携带 RunContext 及其来源说明。
    上游: 三段 if/elif/else 裸字符串赋值，来源说明只通过 echo 临时打印。
    改写: 不可变值对象，source_label 保留决策历史，可审计。
    """
    context: RunContext
    source_label: str    # "CLI" | "env:RAPIDS_RUN_CONTEXT" | "default"


class RunContextResolver:
    """
    封装上游优先级链: CLI > RAPIDS_RUN_CONTEXT 环境变量 > 默认 main。

    上游 bash:
        if [[ -n "${CLI_RUN_CONTEXT:-}" ]]; then RUN_CONTEXT="${CLI_RUN_CONTEXT}"
        elif [[ -n "${RAPIDS_RUN_CONTEXT:-}" ]]; then RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
        else RUN_CONTEXT="main"
        fi

    改写: 三个命名工厂方法 + resolve() 统一入口，优先级链可单独测试。
    """

    @staticmethod
    def from_cli(value: Optional[str]) -> Optional[RunContextSource]:
        """CLI 参数路径（最高优先级）。"""
        if not value:
            return None
        _dbg("RunContextResolver.cli", f"raw={value!r}")
        ctx = RunContextResolver._validate(value, origin="CLI")
        result = RunContextSource(context=ctx, source_label="CLI")
        _dbg("RunContextResolver.cli.ok", f"context={ctx.value}")
        return result

    @staticmethod
    def from_env() -> Optional[RunContextSource]:
        """环境变量路径（中优先级）。"""
        value = os.environ.get("RAPIDS_RUN_CONTEXT", "")
        if not value:
            return None
        _dbg("RunContextResolver.env", f"RAPIDS_RUN_CONTEXT={value!r}")
        ctx = RunContextResolver._validate(value, origin="env:RAPIDS_RUN_CONTEXT")
        result = RunContextSource(context=ctx, source_label="env:RAPIDS_RUN_CONTEXT")
        _dbg("RunContextResolver.env.ok", f"context={ctx.value}")
        return result

    @staticmethod
    def default() -> RunContextSource:
        """默认路径（最低优先级），对应上游 RUN_CONTEXT="main"。"""
        _dbg("RunContextResolver.default", "falling back to main")
        return RunContextSource(context=RunContext.MAIN, source_label="default")

    @staticmethod
    def resolve(cli_context: Optional[str] = None) -> RunContextSource:
        """
        按优先级链解析，返回最终 RunContextSource。
        对应上游整段 if/elif/else + validate 逻辑。
        """
        result = (
            RunContextResolver.from_cli(cli_context)
            or RunContextResolver.from_env()
            or RunContextResolver.default()
        )
        _dbg("RunContextResolver.resolved",
             f"context={result.context.value} source={result.source_label!r}")
        return result

    @staticmethod
    def _validate(value: str, origin: str) -> RunContext:
        if value not in VALID_CONTEXTS:
            raise ValueError(
                f"[RunContextResolver] 无效 run-context {value!r} (来自 {origin})。\n"
                f"可用选项: {sorted(VALID_CONTEXTS)}"
            )
        return RunContext(value)


# ─── 4. BranchNameStrategy — 两路分支名计算 ──────────────────────────────────

@dataclass(frozen=True)
class BranchNameResult:
    """
    封装分支名计算结果。
    上游: 两路内联 if/elif 无名赋值 RAPIDS_BRANCH_NAME + echo 描述。
    改写: 具名二元组，description 保留历史说明，可审计。
    """
    branch_name: str
    description: str


class BranchNameStrategy:
    """
    封装 958783c 的核心分支名决策逻辑:
      main    → branch_name = "main"
      release → branch_name = "release/{YY.MM}"

    上游 bash:
        if [[ "${RUN_CONTEXT}" == "main" ]]; then
            RAPIDS_BRANCH_NAME="main"
            echo "Preparing development branch update ..."
        elif [[ "${RUN_CONTEXT}" == "release" ]]; then
            RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
            echo "Preparing release branch update ..."
        fi

    改写: 静态 compute() 方法，两路各有命名子方法，可独立单元测试。
    """

    @staticmethod
    def compute(ctx: RunContext, ver: VersionSpec) -> BranchNameResult:
        _dbg("BranchNameStrategy.compute", f"ctx={ctx.value} ver={ver.full_tag}")
        if ctx == RunContext.MAIN:
            result = BranchNameStrategy._main_branch(ver)
        else:
            result = BranchNameStrategy._release_branch(ver)
        _dbg("BranchNameStrategy.ok",
             f"branch_name={result.branch_name!r} desc={result.description!r}")
        return result

    @staticmethod
    def _main_branch(ver: VersionSpec) -> BranchNameResult:
        """对应上游 main 路径: RAPIDS_BRANCH_NAME="main"。"""
        return BranchNameResult(
            branch_name="main",
            description=(
                f"Preparing development branch update "
                f"=> {ver.full_tag} (targeting main branch)"
            ),
        )

    @staticmethod
    def _release_branch(ver: VersionSpec) -> BranchNameResult:
        """对应上游 release 路径: RAPIDS_BRANCH_NAME="release/{YY.MM}"。"""
        branch = f"release/{ver.short_tag}"
        return BranchNameResult(
            branch_name=branch,
            description=(
                f"Preparing release branch update "
                f"=> {ver.full_tag} (targeting {branch} branch)"
            ),
        )


# ─── 5. DocRefUpdatePolicy — 文档引用更新策略 ─────────────────────────────────

class DocRefUpdateAction(Enum):
    """对应上游 Documentation references - context-aware 两路行为。"""
    NOOP = "noop"           # main 上下文: 保持外部文档引用为 main，无需变更
    UPDATE_TO_RELEASE = "update_to_release"  # release 上下文: sed 替换为 release/YY.MM


@dataclass(frozen=True)
class DocRefUpdatePolicy:
    """
    封装 cpp/scripts/run-cmake-format.sh 链接更新策略。

    上游 bash:
        if [[ "${RUN_CONTEXT}" == "main" ]]; then
          : # no changes needed
        elif [[ "${RUN_CONTEXT}" == "release" ]]; then
          sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" cpp/scripts/run-cmake-format.sh
        fi

    改写: 值对象携带 action + reason，可在 Python 层验证决策，无需真实 sed 调用。
    注: Walpurgis 无 cmake 体系，此策略仅作文档化迁移，不执行 sed。
    """
    action: DocRefUpdateAction
    reason: str
    target_branch: Optional[str] = None    # release 路径时非 None

    def describe(self) -> str:
        """人类可读的策略描述，对应上游的 echo 输出。"""
        if self.action == DocRefUpdateAction.NOOP:
            return f"DocRef: NOOP — {self.reason}"
        return f"DocRef: UPDATE cpp/scripts/run-cmake-format.sh → {self.target_branch} — {self.reason}"

    @staticmethod
    def for_context(ctx: RunContext, ver: VersionSpec) -> "DocRefUpdatePolicy":
        _dbg("DocRefUpdatePolicy.for_context", f"ctx={ctx.value}")
        if ctx == RunContext.MAIN:
            policy = DocRefUpdatePolicy(
                action=DocRefUpdateAction.NOOP,
                reason="keep external documentation on main (no changes needed)",
            )
        else:
            policy = DocRefUpdatePolicy(
                action=DocRefUpdateAction.UPDATE_TO_RELEASE,
                reason=f"use release branch for external doc links (958783c)",
                target_branch=f"release/{ver.short_tag}",
            )
        _dbg("DocRefUpdatePolicy.ok", policy.describe())
        return policy


# ─── 6. RunContextPolicy — 顶层策略对象 ──────────────────────────────────────

@dataclass(frozen=True)
class RunContextPolicy:
    """
    顶层策略对象，汇总 958783c 引入的所有上下文感知决策。

    上游: 四个散落 bash 变量 (RUN_CONTEXT / RAPIDS_BRANCH_NAME / NEXT_SHORT_TAG / pep440)
         + 三处 if/elif 内联逻辑，无统一结构。
    改写: 单一不可变值对象，所有派生字段均通过类型安全的方式访问。
    """
    version: VersionSpec
    ctx_source: RunContextSource
    branch_result: BranchNameResult
    doc_policy: DocRefUpdatePolicy

    # ── 便利属性（对应上游最常引用的变量）──────────────────────────────────────

    @property
    def run_context(self) -> RunContext:
        return self.ctx_source.context

    @property
    def branch_name(self) -> str:
        """RAPIDS_BRANCH_NAME 等价。"""
        return self.branch_result.branch_name

    @property
    def pep440_norm(self) -> str:
        """NEXT_SHORT_TAG_PEP440 等价。"""
        return self.version.pep440

    @property
    def source_label(self) -> str:
        """决策来源，用于审计日志。"""
        return self.ctx_source.source_label

    def dump(self) -> None:
        """打印完整策略摘要，对应上游各 echo 输出的集合版本。"""
        print(f"[RunContextPolicy] version={self.version.full_tag!r}")
        print(f"[RunContextPolicy] run_context={self.run_context.value!r} (source={self.source_label!r})")
        print(f"[RunContextPolicy] branch_name={self.branch_name!r}")
        print(f"[RunContextPolicy] pep440_norm={self.pep440_norm!r}")
        print(f"[RunContextPolicy] {self.branch_result.description}")
        print(f"[RunContextPolicy] {self.doc_policy.describe()}")


# ─── 7. 顶层工厂函数 ──────────────────────────────────────────────────────────

def resolve_run_context_policy(
    new_version: str,
    cli_context: Optional[str] = None,
) -> RunContextPolicy:
    """
    顶层工厂，对应上游 update-version.sh 的完整决策链。

    参数:
        new_version:  版本字符串，格式 YY.MM.PP，例如 "25.12.00"
        cli_context:  --run-context=... 的值，或 None（走环境变量 / 默认）

    返回:
        RunContextPolicy — 汇总所有决策的不可变策略对象

    示例:
        policy = resolve_run_context_policy("25.12.00", cli_context="release")
        print(policy.branch_name)   # "release/25.12"
        print(policy.pep440_norm)   # "25.12"
    """
    _dbg("resolve.enter", f"new_version={new_version!r} cli_context={cli_context!r}")

    ver = VersionSpec(full_tag=new_version)
    ctx_source = RunContextResolver.resolve(cli_context)
    branch_result = BranchNameStrategy.compute(ctx_source.context, ver)
    doc_policy = DocRefUpdatePolicy.for_context(ctx_source.context, ver)

    policy = RunContextPolicy(
        version=ver,
        ctx_source=ctx_source,
        branch_result=branch_result,
        doc_policy=doc_policy,
    )

    _dbg("resolve.done",
         f"branch_name={policy.branch_name!r} pep440={policy.pep440_norm!r}")

    if _DBG:
        policy.dump()

    return policy


# ─── 8. 自测 ──────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """5项断言覆盖 958783c 核心路径，PASS 即可直接运行本文件验证。"""

    # 测试1: main 路径（CLI 指定）
    p = resolve_run_context_policy("25.12.00", cli_context="main")
    assert p.branch_name == "main", f"test1 failed: {p.branch_name!r}"
    assert p.pep440_norm == "25.12", f"test1 pep440 failed: {p.pep440_norm!r}"
    assert p.doc_policy.action.value == "noop"
    print("[PASS] test1: main 路径 CLI 指定")

    # 测试2: release 路径（CLI 指定）
    p = resolve_run_context_policy("25.12.00", cli_context="release")
    assert p.branch_name == "release/25.12", f"test2 failed: {p.branch_name!r}"
    assert p.doc_policy.target_branch == "release/25.12"
    assert p.doc_policy.action.value == "update_to_release"
    print("[PASS] test2: release 路径 CLI 指定")

    # 测试3: 默认路径（无 CLI，无环境变量）
    old = os.environ.pop("RAPIDS_RUN_CONTEXT", None)
    try:
        p = resolve_run_context_policy("26.02.00", cli_context=None)
        assert p.branch_name == "main"
        assert p.source_label == "default"
    finally:
        if old is not None:
            os.environ["RAPIDS_RUN_CONTEXT"] = old
    print("[PASS] test3: 默认 main 路径")

    # 测试4: 环境变量路径（中优先级）
    os.environ["RAPIDS_RUN_CONTEXT"] = "release"
    try:
        p = resolve_run_context_policy("26.02.00", cli_context=None)
        assert p.branch_name == "release/26.02"
        assert p.source_label == "env:RAPIDS_RUN_CONTEXT"
    finally:
        del os.environ["RAPIDS_RUN_CONTEXT"]
    print("[PASS] test4: 环境变量路径")

    # 测试5: 无效 run-context → ValueError
    raised = False
    try:
        resolve_run_context_policy("25.12.00", cli_context="invalid_ctx")
    except ValueError as e:
        raised = True
        assert "invalid_ctx" in str(e)
    assert raised, "test5: 应该抛出 ValueError"
    print("[PASS] test5: 无效 context 拒绝")

    print("\n[ALL PASS] 958783c smoke test 全部通过")


if __name__ == "__main__":
    _smoke_test()


# =============================================================================
# migrate adce20b: Apply suggestion from @greptile-apps[bot]
# =============================================================================
# 上游 commit adce20b8c4c83893bcf7ffcd263ee536337a03c8
# Author: Nate Rock <rockhowse@gmail.com>
# Co-authored-by: greptile-apps[bot] <165735046+greptile-apps[bot]@users.noreply.github.com>
# Date:   Wed Nov 12 08:30:59 2025 -0600
#
# 上游变更 (1 file changed, 1 deletion):
#   ci/release/update-version.sh — 删除重复的 elif 分支（贴错代码导致两个相同的
#   `elif [[ "${RUN_CONTEXT}" == "release" ]]; then` 连续出现）：
#
#   BEFORE (父 commit):
#     elif [[ "${RUN_CONTEXT}" == "release" ]]; then   ← 重复行，由合并冲突/粘贴引入
#     elif [[ "${RUN_CONTEXT}" == "release" ]]; then
#       sed_runner "s|\bmain\b|release/...|g" ...
#     fi
#
#   AFTER (adce20b):
#     elif [[ "${RUN_CONTEXT}" == "release" ]]; then   ← 只保留一条
#       sed_runner "s|\bmain\b|release/...|g" ...
#     fi
#
# CI/merge → SKIP:
#   ci/release/update-version.sh — SKIP：Walpurgis 无 RAPIDS CI release 体系
#
# 迁移位置: src/walpurgis/core/upstream_version_updater.py（本段追加）
#
# 鲁迅拿法改写（≥20%）：
#   1. BranchDispatcher 类 — 将上游 bash if/elif/elif 链强类型化，防止重复分支
#      静默失效；上游无此保护，重复 elif 直到 greptile bot 发现才知道
#   2. DuplicateBranchError — 专用异常，在构造期即拒绝重复上下文注册；上游
#      bash 中重复 elif 会导致第二条永远不执行，故障无声无息
#   3. dispatch_documentation_sed() — 将 sed_runner 调用语义建模为纯 Python，
#      携带 pattern/replacement/target_file 三元组；上游只有裸 bash 字符串
#   4. WALPURGIS_DEBUG=1 断点 — 打印派发决策链（上下文→分支名→sed指令），
#      上游无任何诊断输出
# =============================================================================

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class DuplicateBranchError(ValueError):
    """注册了重复的 elif 分支；adce20b 修复的正是这个隐性 bug 的 Python 等价物。

    上游 bash 脚本中两条相同的 `elif [[ "${RUN_CONTEXT}" == "release" ]]; then`
    导致第二条永远不可达，greptile-apps[bot] 提交建议删除。本类在 Python 层将
    "重复分支"变为显式错误，而非静默忽略。
    """
    def __init__(self, context_key: str, existing_label: str) -> None:
        super().__init__(
            f"[adce20b] 重复的 elif 分支 '{context_key}' 已注册为 '{existing_label}'；"
            f"上游 bash 中此问题导致第二条分支永远不可达（静默失效），"
            f"BranchDispatcher 在构造期即拒绝。"
        )
        self.context_key = context_key
        self.existing_label = existing_label


@dataclass
class SedInstruction:
    """将 bash `sed_runner` 调用建模为 Python 数据——上游只有裸字符串。

    Attributes:
        pattern:      sed 正则模式（bash 侧为 's|...|...|g' 第一段）
        replacement:  替换目标（可含 ${NEXT_SHORT_TAG} 等变量占位符）
        target_file:  受影响文件路径（相对于仓库根）
        flags:        额外 sed 标志（默认 'g' 全局替换）
    """
    pattern: str
    replacement: str
    target_file: str
    flags: str = "g"

    def as_sed_expr(self, next_short_tag: str = "XX.YY") -> str:
        """展开变量占位符，返回可执行 sed 表达式字符串（仅用于诊断/日志）。"""
        rep = self.replacement.replace("${NEXT_SHORT_TAG}", next_short_tag)
        return f"s|{self.pattern}|{rep}|{self.flags}"

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("SedInstruction.pattern 不能为空")
        if not self.target_file:
            raise ValueError("SedInstruction.target_file 不能为空")


class RunContext(str, Enum):
    """RAPIDS update-version.sh 的两种运行上下文。

    adce20b 修复的 elif 重复问题恰好发生在 main/release 分支判断处；
    强枚举防止字符串拼写错误，同时使穷举检查成为可能。
    """
    MAIN = "main"
    RELEASE = "release"


@dataclass
class BranchDispatcher:
    """将 bash if/elif 链建模为可枚举、可验证的 Python 结构。

    上游 ci/release/update-version.sh 中 adce20b 之前存在：
        if   [[ "${RUN_CONTEXT}" == "main"    ]]; then ...
        elif [[ "${RUN_CONTEXT}" == "release" ]]; then   # ← 第1条（正确）
        elif [[ "${RUN_CONTEXT}" == "release" ]]; then   # ← 第2条（重复，adce20b删除）
            sed_runner "s|\\bmain\\b|release/...|g" ...
        fi

    BranchDispatcher 在 register_branch() 时检测重复，adce20b 的修复
    在 Python 层得到结构性保障而非依赖 code review / bot 建议。
    """
    _branches: Dict[RunContext, Tuple[str, List[SedInstruction]]] = field(
        default_factory=dict, init=False
    )

    def register_branch(
        self,
        context: RunContext,
        label: str,
        instructions: List[SedInstruction],
    ) -> "BranchDispatcher":
        """注册一个 elif 分支。重复注册即抛 DuplicateBranchError（adce20b 所修复）。

        Args:
            context:      RunContext 枚举值（MAIN 或 RELEASE）
            label:        人类可读标签，用于诊断输出
            instructions: 该分支下的 sed 指令列表

        Returns:
            self（支持链式调用）

        Raises:
            DuplicateBranchError: 若 context 已被注册过（即上游重复 elif 的 Python 等价）
        """
        _debug = os.environ.get("WALPURGIS_DEBUG", "0") == "1"
        if context in self._branches:
            existing_label = self._branches[context][0]
            if _debug:
                print(
                    f"[DEBUG adce20b] DuplicateBranchError: context={context.value!r} "
                    f"already registered as {existing_label!r} — 即 bash 中重复的 elif"
                )
            raise DuplicateBranchError(context.value, existing_label)
        self._branches[context] = (label, instructions)
        if _debug:
            print(
                f"[DEBUG adce20b] register_branch: context={context.value!r} "
                f"label={label!r} instructions_count={len(instructions)}"
            )
        return self

    def dispatch(
        self,
        run_context_str: str,
        next_short_tag: str,
    ) -> List[SedInstruction]:
        """按 run_context_str 派发，返回应执行的 sed 指令列表。

        Args:
            run_context_str: "main" 或 "release"（来自 CLI / 环境变量）
            next_short_tag:  如 "26.04"（由 update-version.sh 计算）

        Returns:
            该上下文下应执行的 SedInstruction 列表（可为空）

        Raises:
            ValueError: 若 run_context_str 不在已注册的分支中
        """
        _debug = os.environ.get("WALPURGIS_DEBUG", "0") == "1"
        try:
            ctx = RunContext(run_context_str)
        except ValueError:
            valid = [c.value for c in self._branches]
            raise ValueError(
                f"[adce20b] 未知 run_context={run_context_str!r}，"
                f"有效值：{valid}"
            )
        if ctx not in self._branches:
            if _debug:
                print(
                    f"[DEBUG adce20b] dispatch: context={run_context_str!r} "
                    f"→ no branch registered, returning []"
                )
            return []
        label, instructions = self._branches[ctx]
        if _debug:
            print(
                f"[DEBUG adce20b] dispatch: context={run_context_str!r} "
                f"→ branch={label!r}, {len(instructions)} instruction(s)"
            )
            for instr in instructions:
                print(
                    f"[DEBUG adce20b]   sed: {instr.as_sed_expr(next_short_tag)} "
                    f"→ {instr.target_file}"
                )
        return instructions

    def registered_contexts(self) -> List[str]:
        """返回已注册的所有上下文键（用于枚举验证）。"""
        return [ctx.value for ctx in self._branches]


# ---------------------------------------------------------------------------
# adce20b 的具体迁移实例 — 上游 update-version.sh 文档链接分支
# ---------------------------------------------------------------------------

def _build_doc_link_dispatcher(next_short_tag: str = "26.04") -> BranchDispatcher:
    """构建与 adce20b 后等价的文档链接 BranchDispatcher。

    上游逻辑（adce20b 修复后）：
        if   main:    echo "Keeping external documentation references on main branch"
        elif release: sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" cpp/scripts/run-cmake-format.sh
        fi

    此函数将上述逻辑强类型化，并在 MAIN 分支注册一个空指令列表（"保持不变"），
    在 RELEASE 分支注册实际 sed 指令。
    """
    _debug = os.environ.get("WALPURGIS_DEBUG", "0") == "1"
    if _debug:
        print(
            f"[DEBUG adce20b] _build_doc_link_dispatcher: "
            f"next_short_tag={next_short_tag!r}"
        )
    dispatcher = BranchDispatcher()
    # main 分支：保持外部文档引用在 main，无 sed 操作
    dispatcher.register_branch(
        context=RunContext.MAIN,
        label="keep-docs-on-main",
        instructions=[],
    )
    # release 分支：将文档链接中的 'main' 替换为 'release/YY.MM'
    dispatcher.register_branch(
        context=RunContext.RELEASE,
        label="redirect-docs-to-release",
        instructions=[
            SedInstruction(
                pattern=r"\bmain\b",
                replacement=f"release/{next_short_tag}",
                target_file="cpp/scripts/run-cmake-format.sh",
                flags="g",
            )
        ],
    )
    return dispatcher


# adce20b 模块级单例（next_short_tag 用占位符，实际使用时重新构造）
ADCE20B_DOC_DISPATCHER: BranchDispatcher = _build_doc_link_dispatcher("XX.YY")


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _smoke_test_adce20b() -> None:
    """adce20b 迁移自测 — 验证重复分支检测 + 正常派发两条路径。"""
    import os as _os

    # 测试1: main 上下文 → 空指令列表
    d = _build_doc_link_dispatcher("26.04")
    instrs = d.dispatch("main", "26.04")
    assert instrs == [], f"test1: main 应返回空列表，实际={instrs}"
    print("[PASS] test1: main 上下文 → 空指令（保持文档在 main）")

    # 测试2: release 上下文 → 1条 sed 指令
    instrs = d.dispatch("release", "26.04")
    assert len(instrs) == 1, f"test2: release 应返回 1 条指令，实际={len(instrs)}"
    assert instrs[0].target_file == "cpp/scripts/run-cmake-format.sh"
    assert instrs[0].pattern == r"\bmain\b"
    assert "26.04" in instrs[0].as_sed_expr("26.04")
    print("[PASS] test2: release 上下文 → sed s|\\bmain\\b|release/26.04|g 正确")

    # 测试3: 重复注册同一上下文 → DuplicateBranchError（adce20b 核心修复）
    d2 = BranchDispatcher()
    d2.register_branch(RunContext.RELEASE, "first-release", [])
    raised = False
    try:
        d2.register_branch(RunContext.RELEASE, "duplicate-release", [])
    except DuplicateBranchError as e:
        raised = True
        assert "release" in str(e)
        assert "first-release" in str(e)
    assert raised, "test3: 重复注册应抛出 DuplicateBranchError"
    print("[PASS] test3: 重复 elif 分支 → DuplicateBranchError 即时拒绝")

    # 测试4: 未知 run_context → ValueError
    raised = False
    try:
        d.dispatch("staging", "26.04")
    except ValueError as e:
        raised = True
        assert "staging" in str(e)
    assert raised, "test4: 未知 context 应抛出 ValueError"
    print("[PASS] test4: 未知 context → ValueError")

    # 测试5: registered_contexts() 枚举正确
    ctxs = d.registered_contexts()
    assert set(ctxs) == {"main", "release"}, f"test5: 实际={ctxs}"
    print("[PASS] test5: registered_contexts() = ['main', 'release']")

    # 测试6: DEBUG 模式不抛异常（仅验证不崩溃）
    _os.environ["WALPURGIS_DEBUG"] = "1"
    try:
        d3 = _build_doc_link_dispatcher("26.06")
        d3.dispatch("release", "26.06")
    finally:
        del _os.environ["WALPURGIS_DEBUG"]
    print("[PASS] test6: WALPURGIS_DEBUG=1 调试路径正常运行")

    # 测试7: SedInstruction 空 pattern 应拒绝
    raised = False
    try:
        SedInstruction(pattern="", replacement="x", target_file="foo.sh")
    except ValueError:
        raised = True
    assert raised, "test7: SedInstruction 空 pattern 应拒绝"
    print("[PASS] test7: SedInstruction 空 pattern → ValueError")

    print("\n[ALL PASS] adce20b smoke test 全部通过（7/7）")
