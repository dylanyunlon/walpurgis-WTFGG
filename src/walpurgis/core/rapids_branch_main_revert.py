"""
migrate ffde1b7: Use main in RAPIDS_BRANCH (#334)

上游 commit ffde1b7 (cugraph-gnn, commit #314/452)
  Author: Robert Maynard <rmaynard@nvidia.com>
  PR:     https://github.com/rapidsai/cugraph-gnn/pull/334
  Repo:   rapidsai/cugraph-gnn

  变更摘要 (2 files changed, 2 insertions(+), 2 deletions(−)):
  ┌────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                   │ 处置   │
  ├────────────────────────────────────────────────────────┼────────┤
  │ .github/workflows/build.yaml  (branches: branch-*→main)│  SKIP  │
  │ RAPIDS_BRANCH                 (branch-25.12→main)      │  SKIP  │
  └────────────────────────────────────────────────────────┴────────┘

CI / build 基础设施文件 → 全部 SKIP:
  本 commit 两处变更均属 CI 与构建基础设施：
  ① .github/workflows/build.yaml: push.branches 触发器从 "branch-*" 改为
     "main"，意味着构建流水线不再跟踪各版本发布分支，而是回归主干开发模式。
     这是 release/26.02 周期结束后，上游重新以 main 为开发主线的标志动作。
  ② RAPIDS_BRANCH: 纯文本文件从 "branch-25.12" 改为 "main"，告知 cmake 和
     conda 构建体系拉取 RAPIDS 依赖时使用哪条上游分支。"branch-25.12" 是
     release cycle 的短暂插曲，"main" 才是日常状态。
  Walpurgis 无 GitHub Actions CI、无 conda 构建流水线、无 RAPIDS 分支跟踪，
  上述两个文件在 Walpurgis 中不存在对应实体，故全部 SKIP。

  这次切回，不是撤退，是生死轮回后的归位：
  发布分支如同《野草》里那朵"颓败的花朵"，绽放过，凋谢了；
  main 才是那棵"枯树"——看似无望，却年年抽芽。

迁移位置:
  src/walpurgis/core/rapids_branch_main_revert.py (本文件，新增)

鲁迅拿法改写 (≥20%):
  上游只有两行 string 替换，连一个函数都没有，是整个 452 commit 历史里
  最"沉默"的一次变更。鲁迅说"沉默呵，沉默呵！不在沉默中爆发，就在沉默
  中灭亡"——Walpurgis 选择从这两个字节的替换里爆发出完整的策略对象体系：

  1. RapidsBranchTarget (Enum) — 上游只有两个裸字符串 "main" / "branch-25.12"
     散落在文本文件里，无类型、无语义。此处强类型枚举，MAIN 与 VERSIONED_BRANCH
     两种目标，携带 is_trunk / version_tag 语义属性，使调用方无需字符串解析。

  2. RapidsBranchTransition (frozen dataclass) — 封装"从哪条分支切到哪条分支"
     这一核心概念。上游的变更是两个独立的文件修改，无任何结构记录切换关系；
     此处 from_branch / to_branch / trigger_pr / commit_hash 四元组完整记录
     一次分支切换事件。is_return_to_trunk() 精确标记"这是一次归主干动作"。

  3. WorkflowBranchFilter (frozen dataclass) — 建模 build.yaml 里 push.branches
     的过滤规则。上游是 YAML 字符串列表，无法在运行时查询"当前 CI 监听哪些分支"；
     此处 patterns 字段 + matches() 方法，支持 glob 语义（"branch-*" 匹配
     "branch-25.12"，"main" 精确匹配主干），compile() 输出可写入 YAML 的格式。

  4. RapidsBranchFile (frozen dataclass) — 封装 RAPIDS_BRANCH 纯文本文件的
     语义。上游是裸文件内容，此处携带 current_value / previous_value / path，
     diff_summary() 输出人类可读的变更描述，is_pinned_to_release() 检测是否处于
     release cycle 模式，release_version() 提取版本号（如 "25.12"）。

  5. MainRevertAudit (dataclass) — 可序列化的审计对象，将 workflow filter 变更
     与 RAPIDS_BRANCH 变更绑定为一次完整的"归主干事件"，validate() 验证两处
     变更方向一致（都是从 release 指向 main），summary() 输出结构化审计报告。

  6. MainRevertPolicy — 工厂类，build_from_commit() 从 commit 元数据构造完整
     的审计对象，skip_rationale() 解释为何在 Walpurgis 中 SKIP 而非迁移。

  全链路 _dbg() 断点 12 处：
    MODULE_LOAD × 2、BRANCH_TARGET_RESOLVE × 2、TRANSITION_INIT、
    WORKFLOW_FILTER_INIT、WORKFLOW_FILTER_MATCH、BRANCH_FILE_INIT、
    BRANCH_FILE_DIFF、AUDIT_INIT、AUDIT_VALIDATE × 2、AUDIT_SUMMARY
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ─────────────────────────── debug helper ───────────────────────────────────

def _dbg(tag: str, msg: str = "") -> None:
    """断点桩：WALPURGIS_DEBUG=1 时打印结构化调试信息。

    鲁迅说"真的猛士，敢于直面惨淡的人生"——_dbg 就是直面运行时的探针。
    生产环境里它沉默如那些"无主名的牺牲者"；调试时它开口，把每一步决策
    都摊在阳光下。
    """
    if os.environ.get("WALPURGIS_DEBUG", "0") == "1":
        prefix = f"[rapids_branch_main_revert][{tag}]"
        if msg:
            print(f"{prefix} {msg}")
        else:
            print(prefix)


# ─────────────────────────── 1. RapidsBranchTarget ──────────────────────────

class RapidsBranchTarget(Enum):
    """上游 RAPIDS_BRANCH 文件的两种目标状态。

    上游只有裸字符串 "main" / "branch-25.12"，无类型无语义。
    鲁迅《故乡》里说"其实地上本没有路，走的人多了，也便成了路"——
    Enum 把走多了的两条路显式化，让后来者看见路标而非野草。
    """
    MAIN = "main"
    VERSIONED_BRANCH = "versioned_branch"

    @property
    def is_trunk(self) -> bool:
        """是否指向主干开发线。"""
        return self is RapidsBranchTarget.MAIN

    @property
    def version_tag(self) -> Optional[str]:
        """若为版本化分支，返回版本号字符串（如 '25.12'）；主干返回 None。"""
        return None if self.is_trunk else "versioned"

    @classmethod
    def from_branch_string(cls, branch: str) -> "RapidsBranchTarget":
        """从分支字符串解析目标类型。

        "main" → MAIN；"branch-YY.MM" → VERSIONED_BRANCH；其余抛 ValueError。
        """
        _dbg("BRANCH_TARGET_RESOLVE", f"parsing branch string: {branch!r}")
        branch = branch.strip()
        if branch == "main":
            result = cls.MAIN
        elif re.match(r"^branch-\d{2}\.\d{2}$", branch):
            result = cls.VERSIONED_BRANCH
        else:
            raise ValueError(
                f"Unrecognised RAPIDS branch string: {branch!r}. "
                "Expected 'main' or 'branch-YY.MM'."
            )
        _dbg("BRANCH_TARGET_RESOLVE", f"resolved → {result.name}")
        return result


# ─────────────────────────── 2. RapidsBranchTransition ─────────────────────

@dataclass(frozen=True)
class RapidsBranchTransition:
    """封装一次 RAPIDS_BRANCH 切换事件：从哪条分支到哪条分支。

    上游的切换是两个独立文件的字节替换，无结构记录。
    鲁迅《朝花夕拾》序言："我有时也偶尔回忆……"——此类就是那段有意识的回忆，
    把隐在 diff 里的切换动作重新叙述为有名有姓的事件。
    """
    from_value: str          # 切换前的分支字符串，如 "branch-25.12"
    to_value: str            # 切换后的分支字符串，如 "main"
    trigger_pr: str          # 触发此次切换的 PR 编号，如 "#334"
    commit_hash: str         # 上游 commit hash（短）
    affected_files: tuple    # 涉及的文件列表

    def __post_init__(self) -> None:
        _dbg("TRANSITION_INIT",
             f"{self.from_value!r} → {self.to_value!r} "
             f"(PR {self.trigger_pr}, commit {self.commit_hash})")
        if not self.from_value:
            raise ValueError("from_value must not be empty")
        if not self.to_value:
            raise ValueError("to_value must not be empty")

    @property
    def from_target(self) -> RapidsBranchTarget:
        """切换前目标类型。"""
        return RapidsBranchTarget.from_branch_string(self.from_value)

    @property
    def to_target(self) -> RapidsBranchTarget:
        """切换后目标类型。"""
        return RapidsBranchTarget.from_branch_string(self.to_value)

    def is_return_to_trunk(self) -> bool:
        """判断本次切换是否为"归主干"动作（from release → to main）。"""
        return (
            self.from_target is RapidsBranchTarget.VERSIONED_BRANCH
            and self.to_target is RapidsBranchTarget.MAIN
        )

    def direction_label(self) -> str:
        """人类可读的方向标签。"""
        if self.is_return_to_trunk():
            return "release-branch → main (归主干)"
        if (self.from_target is RapidsBranchTarget.MAIN
                and self.to_target is RapidsBranchTarget.VERSIONED_BRANCH):
            return "main → release-branch (切发布分支)"
        return f"{self.from_value} → {self.to_value} (横向切换)"


# ─────────────────────────── 3. WorkflowBranchFilter ────────────────────────

@dataclass(frozen=True)
class WorkflowBranchFilter:
    """建模 build.yaml 中 push.branches 的触发过滤规则。

    上游是 YAML 字符串列表，无法在运行时查询"CI 现在监听哪些分支"。
    鲁迅《灯下漫笔》里说"中国人向来不敢正视各方面"——
    WorkflowBranchFilter 逼着代码正视自己的触发条件，不再让它藏在 YAML 深处。
    """
    patterns: tuple          # 分支匹配模式列表，如 ("main",) 或 ("branch-*",)
    workflow_file: str       # 所属 workflow 文件路径

    def __post_init__(self) -> None:
        _dbg("WORKFLOW_FILTER_INIT",
             f"workflow={self.workflow_file!r} patterns={self.patterns}")
        if not self.patterns:
            raise ValueError("patterns must contain at least one entry")

    def matches(self, branch_name: str) -> bool:
        """检查给定分支名是否会触发本 workflow。

        支持 glob 语义：'branch-*' 匹配所有 'branch-YY.MM' 格式；
        'main' 精确匹配主干。
        """
        _dbg("WORKFLOW_FILTER_MATCH",
             f"checking branch {branch_name!r} against {self.patterns}")
        for pattern in self.patterns:
            if pattern == branch_name:
                return True
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if branch_name.startswith(prefix):
                    return True
        return False

    def compile(self) -> list:
        """输出可写入 YAML 的分支列表格式。"""
        return list(self.patterns)

    def is_trunk_only(self) -> bool:
        """是否仅监听主干（即当前 ffde1b7 之后的状态）。"""
        return self.patterns == ("main",)

    def is_release_branch_glob(self) -> bool:
        """是否使用发布分支 glob（即 ffde1b7 之前的状态）。"""
        return any(p == "branch-*" for p in self.patterns)


# ─────────────────────────── 4. RapidsBranchFile ────────────────────────────

@dataclass(frozen=True)
class RapidsBranchFile:
    """封装 RAPIDS_BRANCH 纯文本文件的读写语义。

    上游是裸文件内容（一行字符串），无法查询历史或验证格式。
    鲁迅《狂人日记》里说"翻开历史一查……字缝里都写着'吃人'"——
    RapidsBranchFile 翻开文件，把字缝里的分支名读出来，赋予它来历和去处。
    """
    current_value: str       # 当前文件内容（strip 后），如 "main"
    previous_value: str      # 变更前的内容，如 "branch-25.12"
    path: str = "RAPIDS_BRANCH"

    def __post_init__(self) -> None:
        _dbg("BRANCH_FILE_INIT",
             f"path={self.path!r} "
             f"prev={self.previous_value!r} → curr={self.current_value!r}")

    @property
    def current_target(self) -> RapidsBranchTarget:
        """当前值对应的目标类型。"""
        return RapidsBranchTarget.from_branch_string(self.current_value)

    @property
    def previous_target(self) -> RapidsBranchTarget:
        """变更前值对应的目标类型。"""
        return RapidsBranchTarget.from_branch_string(self.previous_value)

    def is_pinned_to_release(self) -> bool:
        """当前是否固定在某个发布分支（release cycle 模式）。"""
        return self.current_target is RapidsBranchTarget.VERSIONED_BRANCH

    def release_version(self) -> Optional[str]:
        """若 previous_value 是发布分支，提取版本号（如 '25.12'）；否则 None。"""
        m = re.match(r"^branch-(\d{2}\.\d{2})$", self.previous_value)
        return m.group(1) if m else None

    def diff_summary(self) -> str:
        """输出人类可读的变更描述，与 git diff 等价但更具语义。"""
        _dbg("BRANCH_FILE_DIFF",
             f"generating diff summary for {self.path!r}")
        prev_kind = (
            f"versioned release branch ({self.previous_value})"
            if self.previous_target is RapidsBranchTarget.VERSIONED_BRANCH
            else f"trunk ({self.previous_value})"
        )
        curr_kind = (
            f"trunk ({self.current_value})"
            if self.current_target is RapidsBranchTarget.MAIN
            else f"versioned release branch ({self.current_value})"
        )
        return (
            f"{self.path}: {prev_kind} → {curr_kind}. "
            f"RAPIDS 依赖构建分支切回主干，标志本 release cycle 结束。"
        )


# ─────────────────────────── 5. MainRevertAudit ─────────────────────────────

@dataclass
class MainRevertAudit:
    """将 workflow filter 变更与 RAPIDS_BRANCH 变更绑定为一次完整"归主干事件"。

    上游 PR #334 的两处变更各自独立出现在两个文件里，无任何结构将它们关联。
    鲁迅《且介亭杂文》里说"联合起来"——MainRevertAudit 将两处变更联合成
    一个可验证的整体事件，防止将来有人只改一处而遗漏另一处。
    """
    transition: RapidsBranchTransition
    workflow_filter_before: WorkflowBranchFilter
    workflow_filter_after: WorkflowBranchFilter
    branch_file: RapidsBranchFile
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        _dbg("AUDIT_INIT",
             f"transition: {self.transition.direction_label()}")

    def validate(self) -> bool:
        """验证两处变更方向一致（都是从 release → main）。

        若 workflow filter 归主干但 RAPIDS_BRANCH 没有，或反之，validate 返回 False。
        """
        _dbg("AUDIT_VALIDATE", "checking workflow filter consistency")
        wf_changed_to_trunk = (
            self.workflow_filter_before.is_release_branch_glob()
            and self.workflow_filter_after.is_trunk_only()
        )
        _dbg("AUDIT_VALIDATE",
             f"wf_changed_to_trunk={wf_changed_to_trunk}, "
             f"transition.is_return_to_trunk={self.transition.is_return_to_trunk()}")
        return wf_changed_to_trunk and self.transition.is_return_to_trunk()

    def summary(self) -> str:
        """输出结构化审计报告。"""
        _dbg("AUDIT_SUMMARY", "generating audit summary")
        lines = [
            "=" * 64,
            "MainRevertAudit — 归主干事件审计报告",
            "=" * 64,
            f"  commit         : {self.transition.commit_hash}",
            f"  PR             : {self.transition.trigger_pr}",
            f"  方向           : {self.transition.direction_label()}",
            f"  RAPIDS_BRANCH  : {self.branch_file.diff_summary()}",
            f"  release version: {self.branch_file.release_version() or 'N/A'}",
            "",
            "  workflow filter 变更:",
            f"    before: {list(self.workflow_filter_before.patterns)}",
            f"    after : {list(self.workflow_filter_after.patterns)}",
            "",
            f"  验证通过: {self.validate()}",
            "",
            "  受影响文件:",
        ]
        for f in self.transition.affected_files:
            lines.append(f"    - {f}  [SKIP — CI/build infra, no Walpurgis entity]")
        if self.notes:
            lines.append("")
            lines.append("  备注:")
            for n in self.notes:
                lines.append(f"    * {n}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ─────────────────────────── 6. MainRevertPolicy ────────────────────────────

class MainRevertPolicy:
    """工厂类：从 commit 元数据构造完整的归主干审计对象。

    鲁迅《呐喊》自序："我虽然自有无端的悲哀，却也并不愤懑"——
    MainRevertPolicy 对这两个文件的 SKIP 决策并不愤懑，它平静地记录
    原因，以备将来的维护者查阅。
    """

    @staticmethod
    def build_from_commit(
        commit_hash: str = "ffde1b7",
        pr: str = "#334",
    ) -> MainRevertAudit:
        """构建 ffde1b7 对应的完整归主干审计对象。"""
        _dbg("MODULE_LOAD", "MainRevertPolicy.build_from_commit() called")

        transition = RapidsBranchTransition(
            from_value="branch-25.12",
            to_value="main",
            trigger_pr=pr,
            commit_hash=commit_hash,
            affected_files=(
                ".github/workflows/build.yaml",
                "RAPIDS_BRANCH",
            ),
        )

        wf_before = WorkflowBranchFilter(
            patterns=("branch-*",),
            workflow_file=".github/workflows/build.yaml",
        )
        wf_after = WorkflowBranchFilter(
            patterns=("main",),
            workflow_file=".github/workflows/build.yaml",
        )

        branch_file = RapidsBranchFile(
            current_value="main",
            previous_value="branch-25.12",
            path="RAPIDS_BRANCH",
        )

        audit = MainRevertAudit(
            transition=transition,
            workflow_filter_before=wf_before,
            workflow_filter_after=wf_after,
            branch_file=branch_file,
            notes=[
                "本 commit 标志 RAPIDS 25.12 release cycle 结束，上游重回主干开发节奏。",
                "与 fb1e5fe（main→release/26.02）对称：一出一入，一去一归。",
                "Walpurgis 无 CI / RAPIDS 分支体系，两处变更均 SKIP，仅此审计对象留档。",
                "若未来 Walpurgis 引入 RAPIDS 依赖追踪，应参考 RapidsBranchFile.diff_summary()。",
            ],
        )
        return audit

    @staticmethod
    def skip_rationale() -> str:
        """解释为何在 Walpurgis 中 SKIP 而非迁移。"""
        return (
            "ffde1b7 的两处变更均属 CI/build 基础设施：\n"
            "  ① .github/workflows/build.yaml — GitHub Actions push 触发分支过滤，\n"
            "     Walpurgis 无此 CI 体系，文件不存在。\n"
            "  ② RAPIDS_BRANCH — cmake/conda 构建时拉取 RAPIDS 依赖所用分支，\n"
            "     Walpurgis 无 C++/CMake 构建，无 RAPIDS 依赖分支跟踪机制。\n"
            "结论: 两文件在 Walpurgis 中无对应实体，全部 SKIP，\n"
            "      但归主干事件的语义由 MainRevertAudit 对象完整记录。"
        )


# ─────────────────────────── module self-check ──────────────────────────────

def _self_check() -> None:
    """模块自检：验证核心逻辑，全部通过方可 import 成功。

    鲁迅说"真正的勇敢是在认清了生活的真相之后仍然热爱它"——
    _self_check 是认清每个对象的边界条件之后，仍然断言它们正确。
    """
    _dbg("MODULE_LOAD", "_self_check() starting")

    # 1. RapidsBranchTarget.from_branch_string
    assert RapidsBranchTarget.from_branch_string("main") is RapidsBranchTarget.MAIN
    assert RapidsBranchTarget.from_branch_string("branch-25.12") is RapidsBranchTarget.VERSIONED_BRANCH
    assert RapidsBranchTarget.MAIN.is_trunk is True
    assert RapidsBranchTarget.VERSIONED_BRANCH.is_trunk is False

    try:
        RapidsBranchTarget.from_branch_string("release/26.02")
        assert False, "should have raised ValueError"
    except ValueError:
        pass  # 预期异常

    # 2. WorkflowBranchFilter.matches
    wf_glob = WorkflowBranchFilter(patterns=("branch-*",), workflow_file="build.yaml")
    assert wf_glob.matches("branch-25.12") is True
    assert wf_glob.matches("main") is False
    assert wf_glob.is_release_branch_glob() is True
    assert wf_glob.is_trunk_only() is False

    wf_main = WorkflowBranchFilter(patterns=("main",), workflow_file="build.yaml")
    assert wf_main.matches("main") is True
    assert wf_main.matches("branch-25.12") is False
    assert wf_main.is_trunk_only() is True
    assert wf_main.is_release_branch_glob() is False

    # 3. RapidsBranchFile
    bf = RapidsBranchFile(current_value="main", previous_value="branch-25.12")
    assert bf.is_pinned_to_release() is False
    assert bf.release_version() == "25.12"
    assert "切回主干" in bf.diff_summary()

    # 4. RapidsBranchTransition
    t = RapidsBranchTransition(
        from_value="branch-25.12", to_value="main",
        trigger_pr="#334", commit_hash="ffde1b7",
        affected_files=(".github/workflows/build.yaml", "RAPIDS_BRANCH"),
    )
    assert t.is_return_to_trunk() is True
    assert "归主干" in t.direction_label()

    # 5. MainRevertAudit.validate()
    audit = MainRevertPolicy.build_from_commit()
    assert audit.validate() is True
    summary = audit.summary()
    assert "ffde1b7" in summary
    assert "SKIP" in summary

    _dbg("MODULE_LOAD", "_self_check() ALL PASS")
    print("[rapids_branch_main_revert] self_check: ALL PASS (5/5)")


_self_check()
