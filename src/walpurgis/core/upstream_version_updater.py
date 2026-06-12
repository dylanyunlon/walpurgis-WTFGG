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
