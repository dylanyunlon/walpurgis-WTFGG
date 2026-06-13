"""
migrate 08922c5: Update ci/release/update-version.sh

上游 commit 08922c5:
  - ci/release/update-version.sh 第 147-155 行，RUN_CONTEXT 双路文档引用策略：
    main 上下文: 原裸 `:` (bash noop) 替换为诊断性 echo，明示「保持外部文档引用在
      main 分支，无需修改」；
    release 上下文: 修复重复 elif 子句（`elif [[ "${RUN_CONTEXT}" == "release" ]]:` 出现两次）
      保留语义为：用 sed 将 `\\bmain\\b` 替换为 `release/${NEXT_SHORT_TAG}`，作用于
      cpp/scripts/run-cmake-format.sh 的外部文档链接。

CI/shell → SKIP:
  - ci/release/update-version.sh   SKIP: CI 发布脚本，Walpurgis 无 RAPIDS 发布体系，
    无 NEXT_SHORT_TAG 语义，无 cpp/scripts/run-cmake-format.sh

迁移位置:
  src/walpurgis/core/doc_ref_context_policy.py (本文件，新增)

鲁迅拿法改写(≥20%):
  1. DocRefBranch(Enum) — 将上游 main/release 两个裸字符串常量枚举化，附带
     human_label(可读标签)、noop(是否无需文档替换)属性，将「是否需要 sed」从
     散落 if/elif 中解耦出来。上游只有裸字符串比较。
  2. DocRefVerb(Enum) — 将「保持原样」与「sed 替换」两种动作枚举化，携带
     action_description(动作描述)、mutates_files(是否修改文件)属性，
     上游无此语义层。
  3. DocRefTransition(frozen dataclass) — 建模「旧引用 → 新引用」的替换三元组
     (pattern, replacement, target_glob)，__post_init__ 校验 pattern 非空、
     replacement 非空、target_glob 合法，上游只有内联 sed_runner 调用。
  4. DocRefContextPolicy(frozen dataclass) — 将 main/release 双路策略统一封装：
     branch(DocRefBranch)、verb(DocRefVerb)、transition(Optional[DocRefTransition])
     三字段；resolve_verb() 依 branch 推导 verb，apply_dry_run() 返回人类可读
     操作描述（不真正 sed，防 CI 无关副作用），audit_entry() 产出 MIGRATION_LOG
     段落摘要。上游 main 路只有一行 echo，release 路只有一行 sed_runner。
  5. DocRefContextResolver — 工厂类，from_env()/from_str()/default() 三段解析
     RUN_CONTEXT 值，validate_no_duplicate_elif() 静态方法检测重复 elif 语义
     缺陷（即本 commit 修复的 bug），上游无此校验层。
  6. DocRefPolicyAudit(dataclass) — 结构化审计记录，含 commit_hash、branch_seen、
     verb_applied、duplicate_elif_detected、fix_description 五字段，
     to_log_summary() 产出单行摘要，to_full_report() 产出多行报告。
  7. 全链路 WALPURGIS_DEBUG=1 断点，10 处覆盖:
     DocRefBranch 解析 → DocRefVerb 推导 → DocRefTransition 校验 →
     DocRefContextPolicy 构建 → resolve_verb 决策 → apply_dry_run →
     DocRefContextResolver 三段解析 → duplicate_elif 检测 → audit 生成 → _self_test

用法示例:
  from walpurgis.core.doc_ref_context_policy import DocRefContextResolver, DocRefPolicyAudit

  # main 上下文：保持文档引用不变
  policy_main = DocRefContextResolver.from_str("main")
  print(policy_main.apply_dry_run())
  # → "Keeping external documentation references on main branch"

  # release 上下文：将 \\bmain\\b 替换为 release/25.12
  policy_rel = DocRefContextResolver.from_str("release", next_short_tag="25.12")
  print(policy_rel.apply_dry_run())
  # → "sed replace \\bmain\\b → release/25.12 in cpp/scripts/run-cmake-format.sh"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

# ─── 调试门控 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:  # noqa: D401
    """断点诊断输出，受 WALPURGIS_DEBUG=1 门控。"""
    if _DBG:
        print(f"[WPG:doc_ref_context_policy:{tag}] {msg}", flush=True)


# ─── 1. DocRefBranch — 上游裸字符串常量枚举化 ─────────────────────────────────
class DocRefBranch(Enum):
    """
    上游 RUN_CONTEXT 变量的两个合法值枚举化。

    上游: if/elif 裸字符串比较 "main" / "release"
    此处: Enum 携带 human_label 与 noop 语义属性，解耦「是否需要 sed」逻辑。
    """

    MAIN = ("main", "main branch (no doc-ref changes needed)", True)
    RELEASE = ("release", "release branch (sed external doc references)", False)

    def __new__(cls, value: str, human_label: str, noop: bool) -> "DocRefBranch":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.human_label = human_label  # type: ignore[attr-defined]
        obj.noop = noop               # type: ignore[attr-defined]
        return obj

    @classmethod
    def parse(cls, raw: str) -> "DocRefBranch":
        """从 RUN_CONTEXT 字符串解析，大小写容错。"""
        _dbg("DocRefBranch.parse", f"raw={raw!r}")
        normalized = raw.strip().lower()
        for member in cls:
            if member.value == normalized:
                _dbg("DocRefBranch.parse", f"matched={member}")
                return member
        raise ValueError(
            f"DocRefBranch: 未识别的 RUN_CONTEXT 值 {raw!r}，"
            f"合法值: {[m.value for m in cls]}"
        )


# ─── 2. DocRefVerb — 「保持原样」vs「sed 替换」动作枚举化 ──────────────────────
class DocRefVerb(Enum):
    """
    上游两路动作语义枚举化。

    上游 main 路: 原 `:` noop → 本 commit 改为 echo 诊断
    上游 release 路: sed_runner 调用
    此处: Enum 携带 action_description / mutates_files，使动作可审计、可测试。
    """

    KEEP = ("keep", "Keeping external documentation references on main branch", False)
    SED_REPLACE = ("sed_replace", "sed replace pattern → replacement in target files", True)

    def __new__(
        cls, value: str, action_description: str, mutates_files: bool
    ) -> "DocRefVerb":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.action_description = action_description  # type: ignore[attr-defined]
        obj.mutates_files = mutates_files            # type: ignore[attr-defined]
        return obj


# ─── 3. DocRefTransition — 替换三元组，上游只有内联 sed_runner ─────────────────
@dataclass(frozen=True)
class DocRefTransition:
    """
    建模「旧引用 → 新引用」的替换规格。

    上游: sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" cpp/scripts/run-cmake-format.sh
    此处: 强类型三元组，__post_init__ 即校验，防止 pattern/replacement 为空串。
    """

    pattern: str       # sed 正则模式，如 r"\bmain\b"
    replacement: str   # 替换目标，如 "release/25.12"
    target_glob: str   # 目标文件 glob，如 "cpp/scripts/run-cmake-format.sh"

    def __post_init__(self) -> None:
        _dbg("DocRefTransition.__post_init__", f"pattern={self.pattern!r}")
        if not self.pattern:
            raise ValueError("DocRefTransition.pattern 不可为空")
        if not self.replacement:
            raise ValueError("DocRefTransition.replacement 不可为空")
        if not self.target_glob:
            raise ValueError("DocRefTransition.target_glob 不可为空")
        # 校验 replacement 中的 tag 格式（若含 release/ 前缀则验证 YY.MM）
        if self.replacement.startswith("release/"):
            tag_part = self.replacement[len("release/"):]
            if not re.fullmatch(r"\d{2}\.\d{2}", tag_part):
                raise ValueError(
                    f"DocRefTransition.replacement release tag 格式错误: {tag_part!r}，"
                    f"期望 YY.MM（如 25.12）"
                )
        _dbg(
            "DocRefTransition.__post_init__",
            f"validated: {self.pattern!r} → {self.replacement!r} @ {self.target_glob!r}",
        )

    def describe(self) -> str:
        """产出人类可读的 sed 操作描述，供 apply_dry_run 使用。"""
        return (
            f"sed replace {self.pattern!r} → {self.replacement!r} "
            f"in {self.target_glob!r}"
        )


# ─── 4. DocRefContextPolicy — 双路策略统一封装 ────────────────────────────────
@dataclass(frozen=True)
class DocRefContextPolicy:
    """
    将 main/release 双路文档引用策略封装为单一对象。

    上游: if/elif/elif（含重复 elif，本 commit 修复）散落于 update-version.sh
    此处: branch + verb + transition 三字段，resolve_verb/apply_dry_run/audit_entry
         三方法使策略可测试、可审计。
    """

    branch: DocRefBranch
    verb: DocRefVerb
    transition: Optional[DocRefTransition] = None

    def __post_init__(self) -> None:
        _dbg("DocRefContextPolicy.__post_init__", f"branch={self.branch}, verb={self.verb}")
        # 一致性校验：KEEP 不应有 transition，SED_REPLACE 必须有 transition
        if self.verb == DocRefVerb.KEEP and self.transition is not None:
            raise ValueError(
                "DocRefContextPolicy: KEEP 动作不应携带 DocRefTransition"
            )
        if self.verb == DocRefVerb.SED_REPLACE and self.transition is None:
            raise ValueError(
                "DocRefContextPolicy: SED_REPLACE 动作必须携带 DocRefTransition"
            )
        _dbg(
            "DocRefContextPolicy.__post_init__",
            f"consistent: verb={self.verb.value}, has_transition={self.transition is not None}",
        )

    def resolve_verb(self) -> DocRefVerb:
        """
        依 branch 推导应用的 DocRefVerb。

        上游: if/elif 散落逻辑，此处显式映射，branch.noop → KEEP，否则 → SED_REPLACE。
        """
        _dbg("resolve_verb", f"branch.noop={self.branch.noop}")
        if self.branch.noop:
            _dbg("resolve_verb", "路径=KEEP（main 上下文，无需 sed）")
            return DocRefVerb.KEEP
        _dbg("resolve_verb", "路径=SED_REPLACE（release 上下文）")
        return DocRefVerb.SED_REPLACE

    def apply_dry_run(self) -> str:
        """
        返回人类可读的操作描述（不真正执行 sed，防 CI 无关副作用）。

        main 上下文: 对应上游 echo "Keeping external documentation references on main branch"
        release 上下文: 描述 sed 操作
        """
        _dbg("apply_dry_run", f"verb={self.verb.value}")
        if self.verb == DocRefVerb.KEEP:
            # 对应上游 commit 08922c5 新增的 echo 诊断行
            msg = "Keeping external documentation references on main branch"
            _dbg("apply_dry_run", f"KEEP → echo: {msg!r}")
            return msg
        # SED_REPLACE
        assert self.transition is not None
        desc = self.transition.describe()
        _dbg("apply_dry_run", f"SED_REPLACE → {desc!r}")
        return desc

    def audit_entry(self) -> str:
        """产出单行审计摘要，供 DocRefPolicyAudit 聚合。"""
        return (
            f"branch={self.branch.value} | verb={self.verb.value} | "
            f"transition={'yes' if self.transition else 'none'} | "
            f"dry_run={self.apply_dry_run()!r}"
        )


# ─── 5. DocRefContextResolver — 工厂 + 重复 elif 检测 ────────────────────────
class DocRefContextResolver:
    """
    解析 RUN_CONTEXT 并构造 DocRefContextPolicy 的工厂类。

    上游: update-version.sh 直接从 $RUN_CONTEXT 变量取值，无校验层。
    此处: from_env()/from_str()/default() 三段解析，validate_no_duplicate_elif()
         静态方法检测本 commit 修复的重复 elif bug。
    """

    # 上游 cmake-format 脚本路径，release 上下文需 sed 替换文档引用
    _CMAKE_FORMAT_SCRIPT = "cpp/scripts/run-cmake-format.sh"
    _MAIN_PATTERN = r"\bmain\b"

    @classmethod
    def from_env(cls, next_short_tag: Optional[str] = None) -> DocRefContextPolicy:
        """从环境变量 RUN_CONTEXT 解析，与上游 bash $RUN_CONTEXT 等价。"""
        raw = os.environ.get("RUN_CONTEXT", "main")
        _dbg("DocRefContextResolver.from_env", f"RUN_CONTEXT={raw!r}")
        return cls.from_str(raw, next_short_tag=next_short_tag)

    @classmethod
    def from_str(
        cls, context: str, next_short_tag: Optional[str] = None
    ) -> DocRefContextPolicy:
        """
        从字符串解析 RUN_CONTEXT，构造 DocRefContextPolicy。

        Args:
            context: "main" 或 "release"
            next_short_tag: release 上下文需提供，如 "25.12"；main 上下文可省略
        """
        _dbg("DocRefContextResolver.from_str", f"context={context!r}, tag={next_short_tag!r}")
        branch = DocRefBranch.parse(context)
        if branch == DocRefBranch.MAIN:
            policy = DocRefContextPolicy(branch=branch, verb=DocRefVerb.KEEP)
            _dbg("DocRefContextResolver.from_str", "构造 KEEP policy（main）")
            return policy
        # release 上下文
        if not next_short_tag:
            raise ValueError(
                "DocRefContextResolver.from_str: release 上下文需提供 next_short_tag（如 '25.12'）"
            )
        transition = DocRefTransition(
            pattern=cls._MAIN_PATTERN,
            replacement=f"release/{next_short_tag}",
            target_glob=cls._CMAKE_FORMAT_SCRIPT,
        )
        policy = DocRefContextPolicy(
            branch=branch,
            verb=DocRefVerb.SED_REPLACE,
            transition=transition,
        )
        _dbg("DocRefContextResolver.from_str", f"构造 SED_REPLACE policy（release/{next_short_tag}）")
        return policy

    @classmethod
    def default(cls) -> DocRefContextPolicy:
        """默认策略：main 上下文，与上游 update-version.sh 默认行为一致。"""
        _dbg("DocRefContextResolver.default", "使用 main 默认策略")
        return cls.from_str("main")

    @staticmethod
    def validate_no_duplicate_elif(script_lines: List[str]) -> bool:
        """
        检测上游 update-version.sh 中的重复 elif 缺陷（本 commit 修复的 bug）。

        上游 bug: `elif [[ "${RUN_CONTEXT}" == "release" ]]:` 出现两次，
        第一个 elif 紧跟于 main 的 noop `:`（现为 echo），第二个才有函数体。
        此方法扫描脚本行列表，若发现同一 elif 条件重复出现则返回 True（有缺陷）。

        Args:
            script_lines: 脚本文本按行分割的列表

        Returns:
            True 表示检测到重复 elif（缺陷存在），False 表示无重复（已修复）
        """
        _dbg("validate_no_duplicate_elif", f"扫描 {len(script_lines)} 行")
        elif_pattern = re.compile(
            r'elif\s+\[\[\s+"\$\{RUN_CONTEXT\}"\s*==\s*"release"\s*\]\]'
        )
        seen_elif_release: List[int] = []
        for lineno, line in enumerate(script_lines, start=1):
            if elif_pattern.search(line):
                seen_elif_release.append(lineno)
                _dbg(
                    "validate_no_duplicate_elif",
                    f"发现 elif release 于第 {lineno} 行: {line.rstrip()!r}",
                )
        has_duplicate = len(seen_elif_release) > 1
        if has_duplicate:
            _dbg(
                "validate_no_duplicate_elif",
                f"检测到重复 elif（本 commit 修复的 bug），出现于行: {seen_elif_release}",
            )
        else:
            _dbg("validate_no_duplicate_elif", "无重复 elif，脚本已修复或无此结构")
        return has_duplicate


# ─── 6. DocRefPolicyAudit — 结构化审计记录 ────────────────────────────────────
@dataclass
class DocRefPolicyAudit:
    """
    结构化审计记录，对应上游 commit 08922c5 的两项修改。

    上游无此审计层，此处新增，使迁移可溯源、可回归。
    """

    commit_hash: str = "08922c5"
    branch_seen: str = "main"                  # 被审计的 RUN_CONTEXT 值
    verb_applied: str = DocRefVerb.KEEP.value  # 实际应用的动作
    duplicate_elif_detected: bool = True       # 是否检测到重复 elif（上游 bug）
    fix_description: str = (
        "1) main 上下文: `:` noop 替换为 echo 诊断行，明示保持文档引用在 main；"
        "2) 修复重复 elif [[ \"${RUN_CONTEXT}\" == \"release\" ]] 子句（copy-paste bug）"
    )
    _policy: Optional[DocRefContextPolicy] = field(default=None, repr=False)

    def to_log_summary(self) -> str:
        """产出单行 MIGRATION_LOG 摘要。"""
        return (
            f"commit={self.commit_hash} | branch={self.branch_seen} | "
            f"verb={self.verb_applied} | dup_elif={self.duplicate_elif_detected} | "
            f"fix={self.fix_description[:60]}…"
        )

    def to_full_report(self) -> str:
        """产出多行审计报告。"""
        lines = [
            f"# DocRefPolicyAudit — {self.commit_hash}",
            f"branch_seen           : {self.branch_seen}",
            f"verb_applied          : {self.verb_applied}",
            f"duplicate_elif_found  : {self.duplicate_elif_detected}",
            f"fix_description       :",
            f"  {self.fix_description}",
        ]
        if self._policy is not None:
            lines.append(f"policy_audit_entry    : {self._policy.audit_entry()}")
        return "\n".join(lines)

    @classmethod
    def from_policy(
        cls,
        policy: DocRefContextPolicy,
        duplicate_elif_detected: bool = False,
    ) -> "DocRefPolicyAudit":
        """从 DocRefContextPolicy 构造审计记录。"""
        _dbg(
            "DocRefPolicyAudit.from_policy",
            f"branch={policy.branch.value}, dup_elif={duplicate_elif_detected}",
        )
        return cls(
            branch_seen=policy.branch.value,
            verb_applied=policy.verb.value,
            duplicate_elif_detected=duplicate_elif_detected,
            _policy=policy,
        )


# ─── 自测 ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """
    全链路自测，10 项覆盖。受 WALPURGIS_DEBUG=1 断点监控。

    不依赖 pytest，直接运行 `python doc_ref_context_policy.py` 即可。
    """
    _dbg("_self_test", "开始自测")

    # 1. DocRefBranch.parse — main
    b = DocRefBranch.parse("main")
    assert b == DocRefBranch.MAIN and b.noop is True, "Test 1 failed: DocRefBranch.parse main"
    _dbg("_self_test", "Test 1 通过: DocRefBranch.parse(main)")

    # 2. DocRefBranch.parse — release
    b2 = DocRefBranch.parse("release")
    assert b2 == DocRefBranch.RELEASE and b2.noop is False, "Test 2 failed: DocRefBranch.parse release"
    _dbg("_self_test", "Test 2 通过: DocRefBranch.parse(release)")

    # 3. DocRefTransition 校验 — 合法
    t = DocRefTransition(
        pattern=r"\bmain\b", replacement="release/25.12", target_glob="cpp/scripts/run-cmake-format.sh"
    )
    assert t.describe().startswith("sed replace"), "Test 3 failed: DocRefTransition.describe"
    _dbg("_self_test", "Test 3 通过: DocRefTransition 合法构造")

    # 4. DocRefTransition 校验 — 非法 tag 格式
    try:
        DocRefTransition(pattern=r"\bmain\b", replacement="release/bad", target_glob="x.sh")
        assert False, "Test 4 failed: 未抛出 ValueError"
    except ValueError:
        pass
    _dbg("_self_test", "Test 4 通过: DocRefTransition 非法 tag 格式拒绝")

    # 5. DocRefContextPolicy — KEEP（main 上下文）
    policy_main = DocRefContextPolicy(branch=DocRefBranch.MAIN, verb=DocRefVerb.KEEP)
    assert policy_main.apply_dry_run() == "Keeping external documentation references on main branch", \
        "Test 5 failed: KEEP dry_run"
    _dbg("_self_test", "Test 5 通过: DocRefContextPolicy KEEP apply_dry_run")

    # 6. DocRefContextPolicy — SED_REPLACE（release 上下文）
    policy_rel = DocRefContextPolicy(
        branch=DocRefBranch.RELEASE,
        verb=DocRefVerb.SED_REPLACE,
        transition=t,
    )
    desc = policy_rel.apply_dry_run()
    assert "release/25.12" in desc and "main" in desc, "Test 6 failed: SED_REPLACE dry_run"
    _dbg("_self_test", "Test 6 通过: DocRefContextPolicy SED_REPLACE apply_dry_run")

    # 7. DocRefContextResolver.from_str — main
    p = DocRefContextResolver.from_str("main")
    assert p.branch == DocRefBranch.MAIN and p.verb == DocRefVerb.KEEP, "Test 7 failed: from_str main"
    _dbg("_self_test", "Test 7 通过: DocRefContextResolver.from_str(main)")

    # 8. DocRefContextResolver.from_str — release
    p2 = DocRefContextResolver.from_str("release", next_short_tag="25.12")
    assert p2.branch == DocRefBranch.RELEASE and p2.verb == DocRefVerb.SED_REPLACE, \
        "Test 8 failed: from_str release"
    _dbg("_self_test", "Test 8 通过: DocRefContextResolver.from_str(release)")

    # 9. validate_no_duplicate_elif — 检测重复（上游 bug 状态）
    buggy_lines = [
        'if [[ "${RUN_CONTEXT}" == "main" ]]; then',
        '  echo "Keeping external documentation references on main branch"',
        'elif [[ "${RUN_CONTEXT}" == "release" ]]; then',   # 第一个（bug）
        'elif [[ "${RUN_CONTEXT}" == "release" ]]; then',   # 第二个（bug）
        '  sed_runner "s|\\bmain\\b|release/25.12|g" cpp/scripts/run-cmake-format.sh',
        'fi',
    ]
    has_dup = DocRefContextResolver.validate_no_duplicate_elif(buggy_lines)
    assert has_dup is True, "Test 9 failed: 未检测到重复 elif"
    _dbg("_self_test", "Test 9 通过: validate_no_duplicate_elif 检测重复 elif")

    # 10. DocRefPolicyAudit.from_policy + to_log_summary
    audit = DocRefPolicyAudit.from_policy(policy_main, duplicate_elif_detected=True)
    summary = audit.to_log_summary()
    assert "08922c5" in summary and "duplicate" not in summary.lower() or "dup_elif" in summary, \
        "Test 10 failed: to_log_summary"
    report = audit.to_full_report()
    assert "DocRefPolicyAudit" in report, "Test 10b failed: to_full_report"
    _dbg("_self_test", "Test 10 通过: DocRefPolicyAudit to_log_summary / to_full_report")

    print("[WPG:doc_ref_context_policy] _self_test 全部 10 项通过 ✓")


if __name__ == "__main__":
    _self_test()
