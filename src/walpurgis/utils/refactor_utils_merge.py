"""
migrate 9993ea258: Merge branch 'refactor_utils' into 'master'

上游 Megatron-LM commit #17（共 9062），纯 merge commit，diff 为空。
这是 `refactor_utils` 分支并入主干的集成节点，标志着一轮 utils 重构的终结。

鲁迅笔法：
  一次合并，什么代码都没有动，却是一个时代的结束。
  就像《阿Q正传》里的土地改革——表面上什么都没变，
  但那条线已经画下去了，此后的一切都叫\"主干\"，此前的叫\"分支\"。
  重构不在合并那一刻发生，合并只是宣布：重构已经发生过了。

  `refactor_utils` 分支在上游的使命：将 Megatron-LM 的工具函数从
  散落在各处的临时函数，整理进统一的 utils 模块。
  那些代码没有显示在这个 merge commit 的 diff 里，
  因为它们已经在更早的 commits（被归入此分支）里完成了。
  合并节点本身是空的，空得像一块碑，碑文写的是「已完成」。

  Walpurgis 对应：src/walpurgis/utils/ 已承担了同等职责，
  本文件作为结构化的合并里程碑记录，提供可程序化查询的分支集成元数据。
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    """全局 debug 断点；WALPURGIS_DEBUG=1 时激活，生产环境静默。"""
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG 9993ea258 {tag}] {msg}")


# ── 合并类型枚举 ─────────────────────────────────────────────────────────────

class MergeType:
    """上游 merge commit 类型分类。"""
    FAST_FORWARD = "fast_forward"       # 无交叉历史，直接指针移动
    THREE_WAY = "three_way"             # 有分叉历史，生成 merge commit
    SQUASH = "squash"                   # squash 合并，历史压缩
    EMPTY_DIFF = "empty_diff"           # diff 为空的三方合并（本 commit 属于此类）


# ── 重构分支清单 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RefactoredUtilEntry:
    """
    记录 refactor_utils 分支内一个 utils 重构点的元数据。

    上游 refactor_utils 分支所做的重构不出现在 merge commit 的 diff 中，
    而是分散在分支内更早的提交里。Walpurgis 将其语义结构化记录于此。

    改写点：上游无此数据结构；Walpurgis 新增，使重构历史可程序化查询。
    """
    original_location: str      # 上游重构前的函数/类位置
    refactored_location: str    # 上游重构后的 utils 路径
    walpurgis_path: str         # Walpurgis 对应迁移路径
    description: str            # 重构语义摘要

    def _dbg_entry(self) -> None:
        _dbg(
            "REFACTORED_UTIL_ENTRY",
            f"{self.original_location!r} → {self.refactored_location!r} "
            f"(walpurgis: {self.walpurgis_path!r})"
        )


# ── 合并里程碑 ───────────────────────────────────────────────────────────────

@dataclass
class MergeMilestone:
    """
    结构化记录一个 merge commit 的集成元数据。

    改写点一：上游 merge commit 是 git 对象，无法程序化查询其语义。
              Walpurgis 将其转为可实例化的 Python 对象，支持审计与文档生成。
    改写点二：加入 merge_type 区分合并策略，上游 git log 无此显式标注。
    改写点三：加入 refactored_entries 列表，显式关联被合并分支内的重构内容。
    改写点四：全链路 _dbg() 断点，追踪里程碑实例化路径。
    """
    commit_hash: str
    commit_index: int                           # 在上游 9062 commits 中的序号
    total_commits: int
    source_branch: str                          # 被合并的分支
    target_branch: str                          # 合并目标分支
    commit_subject: str
    merge_type: str = MergeType.EMPTY_DIFF
    diff_files_changed: int = 0
    diff_insertions: int = 0
    diff_deletions: int = 0
    refactored_entries: List[RefactoredUtilEntry] = field(default_factory=list)
    walpurgis_note: str = ""

    def __post_init__(self) -> None:
        _dbg(
            "MILESTONE_INIT",
            f"hash={self.commit_hash!r} idx={self.commit_index}/{self.total_commits} "
            f"type={self.merge_type!r} diff_files={self.diff_files_changed}"
        )

    def is_empty_diff(self) -> bool:
        """判断此 merge commit 是否为零 diff（即 diff 完全为空）。"""
        result = (
            self.diff_files_changed == 0
            and self.diff_insertions == 0
            and self.diff_deletions == 0
        )
        _dbg("IS_EMPTY_DIFF", f"{result}")
        return result

    def integration_summary(self) -> str:
        """
        返回集成摘要字符串，格式化为人类可读的里程碑描述。

        改写点：上游无此方法；Walpurgis 提供统一摘要格式，
        便于 MIGRATION_LOG 自动生成与审计脚本调用。
        """
        _dbg("INTEGRATION_SUMMARY_START", f"hash={self.commit_hash!r}")
        lines = [
            f"MergeMilestone #{self.commit_index}/{self.total_commits}",
            f"  hash        : {self.commit_hash}",
            f"  subject     : {self.commit_subject}",
            f"  merge       : {self.source_branch!r} → {self.target_branch!r}",
            f"  merge_type  : {self.merge_type}",
            f"  diff        : {self.diff_files_changed} files, "
                f"+{self.diff_insertions}/-{self.diff_deletions}",
        ]
        if self.walpurgis_note:
            lines.append(f"  note        : {self.walpurgis_note}")
        if self.refactored_entries:
            lines.append(f"  refactored  : {len(self.refactored_entries)} entries")
            for e in self.refactored_entries:
                lines.append(
                    f"    {e.original_location!r} → {e.walpurgis_path!r}"
                )
        summary = "\n".join(lines)
        _dbg("INTEGRATION_SUMMARY_DONE", f"lines={len(lines)}")
        return summary

    def audit_dict(self) -> dict:
        """
        返回结构化审计字典，供 CI / 自动化脚本解析。

        改写点：上游 git merge commit 无结构化元数据输出；
        Walpurgis 改写为可序列化的 dict，便于 JSON dump 与 diff 对比。
        """
        _dbg("AUDIT_DICT", f"hash={self.commit_hash!r}")
        return {
            "commit_hash": self.commit_hash,
            "commit_index": self.commit_index,
            "total_commits": self.total_commits,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "commit_subject": self.commit_subject,
            "merge_type": self.merge_type,
            "diff": {
                "files_changed": self.diff_files_changed,
                "insertions": self.diff_insertions,
                "deletions": self.diff_deletions,
            },
            "is_empty_diff": self.is_empty_diff(),
            "refactored_entries_count": len(self.refactored_entries),
            "walpurgis_note": self.walpurgis_note,
        }


# ── 实例化本 commit 的里程碑 ─────────────────────────────────────────────────

_dbg("MODULE_LOAD", "初始化 refactor_utils merge milestone 9993ea258")

#: 上游 commit 9993ea258 的结构化里程碑实例
REFACTOR_UTILS_MERGE_MILESTONE = MergeMilestone(
    commit_hash="9993ea258",
    commit_index=17,
    total_commits=9062,
    source_branch="refactor_utils",
    target_branch="master",
    commit_subject="Merge branch 'refactor_utils' into 'master'",
    merge_type=MergeType.EMPTY_DIFF,
    diff_files_changed=0,
    diff_insertions=0,
    diff_deletions=0,
    walpurgis_note=(
        "纯 merge commit，diff 为空。"
        "refactor_utils 分支的实质重构内容分散于该分支内更早的 commits 中，"
        "并非集中于此合并节点。"
        "Walpurgis 对应：src/walpurgis/utils/ 已承担统一工具函数职责，"
        "本文件作为可程序化查询的分支集成元数据记录节点。"
    ),
    refactored_entries=[
        # refactor_utils 分支的重构意图：统一 Megatron-LM 的 utils 工具函数
        # 上游具体重构内容在该分支更早的 commits 内，此处记录语义层面的对应
        RefactoredUtilEntry(
            original_location="megatron/utils.py (散落函数)",
            refactored_location="megatron/utils.py (统一模块)",
            walpurgis_path="src/walpurgis/utils/",
            description=(
                "refactor_utils 分支将 Megatron-LM 散落的工具函数收拢至 utils 模块；"
                "Walpurgis 以 src/walpurgis/utils/ 目录对应此重构语义。"
            ),
        ),
    ],
)

_dbg(
    "MILESTONE_READY",
    f"hash={REFACTOR_UTILS_MERGE_MILESTONE.commit_hash!r} "
    f"empty_diff={REFACTOR_UTILS_MERGE_MILESTONE.is_empty_diff()}"
)


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    运行 5 项断言，验证里程碑数据完整性。

    改写点：上游 merge commit 无可运行的自检逻辑；
    Walpurgis 改写为可在 CI 中调用的自检函数，返回 bool 表示通过与否。
    """
    _dbg("SELF_CHECK_START")
    m = REFACTOR_UTILS_MERGE_MILESTONE

    # 1. hash 格式（9位十六进制）
    assert len(m.commit_hash) == 9 and all(
        c in "0123456789abcdef" for c in m.commit_hash
    ), f"hash 格式非法: {m.commit_hash!r}"
    _dbg("SELF_CHECK_1_HASH", "OK")

    # 2. diff 确实为空
    assert m.is_empty_diff(), "9993ea258 应为 empty diff merge commit"
    _dbg("SELF_CHECK_2_EMPTY_DIFF", "OK")

    # 3. commit index 在合法范围内
    assert 1 <= m.commit_index <= m.total_commits, (
        f"commit_index={m.commit_index} 超出范围 [1, {m.total_commits}]"
    )
    _dbg("SELF_CHECK_3_INDEX", f"index={m.commit_index} OK")

    # 4. source/target branch 非空
    assert m.source_branch and m.target_branch, "source/target branch 不得为空"
    _dbg("SELF_CHECK_4_BRANCHES", f"{m.source_branch!r} → {m.target_branch!r} OK")

    # 5. refactored_entries 至少有一条语义记录
    assert len(m.refactored_entries) >= 1, "应至少有一条 refactored_entry 语义记录"
    _dbg("SELF_CHECK_5_ENTRIES", f"count={len(m.refactored_entries)} OK")

    _dbg("SELF_CHECK_DONE", "全部 5 项断言通过")
    return True


if __name__ == "__main__":
    print(REFACTOR_UTILS_MERGE_MILESTONE.integration_summary())
    print()
    import json
    print(json.dumps(REFACTOR_UTILS_MERGE_MILESTONE.audit_dict(), ensure_ascii=False, indent=2))
    print()
    ok = self_check()
    print(f"self_check: {'PASS' if ok else 'FAIL'}")
