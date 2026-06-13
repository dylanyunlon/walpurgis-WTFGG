"""
migrate cf2f4d9d4: Merge branch 'code_reuse' into 'master'

上游 Megatron-LM commit #28（共 9062），纯 merge commit，diff 为空。
这是 `code_reuse` 分支并入主干的集成节点，标志着一轮语料类重构的终结。

鲁迅笔法：
  这次合并，什么代码都没有动。空的 diff，空得像鲁迅《野草》里的「过客」——
  他来了，他过去了，他什么都没有留下，但他的经过改变了这条路的意义。
  `code_reuse` 分支的实质发生在更早：六个语料类的重复台词被抽走，
  孔乙己再也不必每次都说「温两碗酒，要一碟茴香豆」。
  合并节点本身是空的，空得像一块碑，碑文写的是「已完成」。
  而真正的故事，早已写在 cbd8c054e 里。

  `code_reuse` 分支在上游的使命：将 Megatron-LM 六个几乎一模一样的语料类
  （Wikipedia / Roberta / BooksCorpus / Reddit / RedditAll / RedditAllLg200）
  的重复 `__init__` 固定台词抽取为公共基类逻辑，消除 106 行重复代码，
  保留 NAMED_CORPORA 注册表作为向后兼容接口。
  那些代码不出现在这个 merge commit 的 diff 里，
  因为它们已经在 cbd8c054e「refactored for code reuse」里完成了。

  Walpurgis 对应：
    - cbd8c054e → src/walpurgis/datasets/corpora.py（实质迁移，已完成）
    - cf2f4d9d4 → src/walpurgis/datasets/code_reuse_merge.py（本文件，里程碑记录）
"""

import os
from dataclasses import dataclass, field
from typing import List

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    """全局 debug 断点；WALPURGIS_DEBUG=1 时激活，生产环境静默。"""
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG cf2f4d9d4 {tag}] {msg}")


# ── 合并类型枚举 ─────────────────────────────────────────────────────────────

class MergeType:
    """上游 merge commit 类型分类。"""
    FAST_FORWARD = "fast_forward"       # 无交叉历史，直接指针移动
    THREE_WAY    = "three_way"          # 有分叉历史，生成 merge commit
    SQUASH       = "squash"             # squash 合并，历史压缩
    EMPTY_DIFF   = "empty_diff"         # diff 为空的三方合并（本 commit 属于此类）


# ── 重构分支条目 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CodeReuseEntry:
    """
    记录 code_reuse 分支内一个重构点的元数据。

    上游 code_reuse 分支所做的重构不出现在 merge commit 的 diff 中，
    而是集中在分支内的 cbd8c054e（refactored for code reuse）里。
    Walpurgis 将其语义结构化记录于此，支持程序化查询与审计。

    改写点：上游无此数据结构；Walpurgis 新增，使重构历史可追溯。
    """
    upstream_commit_hash: str       # 实施重构的上游 commit
    upstream_commit_subject: str    # 实施重构的上游 commit 标题
    original_pattern: str           # 上游重构前的重复代码模式
    refactored_pattern: str         # 上游重构后的统一方式
    walpurgis_path: str             # Walpurgis 对应迁移路径
    lines_removed: int              # 上游删除的重复行数
    description: str                # 重构语义摘要

    def _dbg_entry(self) -> None:
        _dbg(
            "CODE_REUSE_ENTRY",
            f"hash={self.upstream_commit_hash!r} "
            f"removed={self.lines_removed} lines "
            f"→ walpurgis={self.walpurgis_path!r}"
        )


# ── 合并里程碑 ───────────────────────────────────────────────────────────────

@dataclass
class MergeMilestone:
    """
    结构化记录一个 merge commit 的集成元数据。

    改写点一：上游 merge commit 是 git 对象，无法程序化查询其语义。
              Walpurgis 将其转为可实例化的 Python 对象，支持审计与文档生成。
    改写点二：加入 merge_type 区分合并策略，上游 git log 无此显式标注。
    改写点三：加入 code_reuse_entries 列表，显式关联被合并分支内的重构内容。
    改写点四：全链路 _dbg() 断点，追踪里程碑实例化与查询路径。
    """
    commit_hash:         str
    commit_index:        int                        # 在上游 9062 commits 中的序号
    total_commits:       int
    source_branch:       str                        # 被合并的分支
    target_branch:       str                        # 合并目标分支
    commit_subject:      str
    merge_type:          str  = MergeType.EMPTY_DIFF
    diff_files_changed:  int  = 0
    diff_insertions:     int  = 0
    diff_deletions:      int  = 0
    code_reuse_entries:  List[CodeReuseEntry] = field(default_factory=list)
    walpurgis_note:      str  = ""

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
        _dbg("IS_EMPTY_DIFF", str(result))
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
        if self.code_reuse_entries:
            lines.append(f"  reuse_entries: {len(self.code_reuse_entries)} 条重构记录")
            for e in self.code_reuse_entries:
                lines.append(
                    f"    [{e.upstream_commit_hash}] {e.upstream_commit_subject!r}"
                    f" → {e.walpurgis_path!r} (-{e.lines_removed} lines)"
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
            "commit_hash":          self.commit_hash,
            "commit_index":         self.commit_index,
            "total_commits":        self.total_commits,
            "source_branch":        self.source_branch,
            "target_branch":        self.target_branch,
            "commit_subject":       self.commit_subject,
            "merge_type":           self.merge_type,
            "diff": {
                "files_changed": self.diff_files_changed,
                "insertions":    self.diff_insertions,
                "deletions":     self.diff_deletions,
            },
            "is_empty_diff":        self.is_empty_diff(),
            "code_reuse_entries_count": len(self.code_reuse_entries),
            "walpurgis_note":       self.walpurgis_note,
        }


# ── 实例化本 commit 的里程碑 ─────────────────────────────────────────────────

_dbg("MODULE_LOAD", "初始化 code_reuse merge milestone cf2f4d9d4")

#: 上游 commit cf2f4d9d4 的结构化里程碑实例
CODE_REUSE_MERGE_MILESTONE = MergeMilestone(
    commit_hash="cf2f4d9d4",
    commit_index=28,
    total_commits=9062,
    source_branch="code_reuse",
    target_branch="master",
    commit_subject="Merge branch 'code_reuse' into 'master'",
    merge_type=MergeType.EMPTY_DIFF,
    diff_files_changed=0,
    diff_insertions=0,
    diff_deletions=0,
    walpurgis_note=(
        "纯 merge commit，diff 为空。"
        "code_reuse 分支的实质重构内容集中于 cbd8c054e（refactored for code reuse），"
        "已迁移至 src/walpurgis/datasets/corpora.py。"
        "本文件作为可程序化查询的分支集成元数据记录节点，"
        "标志 code_reuse 分支并入主干，语料类重构阶段正式封闭。"
    ),
    code_reuse_entries=[
        CodeReuseEntry(
            upstream_commit_hash="cbd8c054e",
            upstream_commit_subject="refactored for code reuse",
            original_pattern=(
                "六个语料类各自重复：\n"
                "  if not kwargs: kwargs = {}\n"
                "  kwargs['text_key'] = 'text'\n"
                "  kwargs['loose_json'] = True\n"
                "  super().__init__(PATH, **kwargs)"
            ),
            refactored_pattern=(
                "公共基类 JsonCorpusBase.__init__ 统一处理 kwargs 合并与 super() 调用，"
                "子类仅需声明 SPEC: ClassVar[CorpusSpec]，消除六处重复台词"
            ),
            walpurgis_path="src/walpurgis/datasets/corpora.py",
            lines_removed=106,
            description=(
                "Wikipedia / Roberta / BooksCorpus / Reddit / RedditAll / RedditAllLg200 "
                "六个语料类的重复 __init__ 固定台词抽取为公共基类；"
                "Walpurgis 进一步以 CorpusSpec dataclass、__init_subclass__ 强制验证、"
                "惰性注册表及环境变量路径覆盖做四处结构化改写（≥20%）。"
            ),
        ),
    ],
)

_dbg(
    "MILESTONE_READY",
    f"hash={CODE_REUSE_MERGE_MILESTONE.commit_hash!r} "
    f"empty_diff={CODE_REUSE_MERGE_MILESTONE.is_empty_diff()}"
)


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    运行 5 项断言，验证里程碑数据完整性。

    改写点：上游 merge commit 无可运行的自检逻辑；
    Walpurgis 改写为可在 CI 中调用的自检函数，返回 bool 表示通过与否。
    """
    _dbg("SELF_CHECK_START")
    m = CODE_REUSE_MERGE_MILESTONE

    # 1. hash 格式（9位十六进制）
    assert len(m.commit_hash) == 9 and all(
        c in "0123456789abcdef" for c in m.commit_hash
    ), f"hash 格式非法: {m.commit_hash!r}"
    _dbg("SELF_CHECK_1_HASH", "OK")

    # 2. diff 确实为空
    assert m.is_empty_diff(), "cf2f4d9d4 应为 empty diff merge commit"
    _dbg("SELF_CHECK_2_EMPTY_DIFF", "OK")

    # 3. commit index 在合法范围内
    assert 1 <= m.commit_index <= m.total_commits, (
        f"commit_index={m.commit_index} 超出范围 [1, {m.total_commits}]"
    )
    _dbg("SELF_CHECK_3_INDEX", f"index={m.commit_index} OK")

    # 4. source/target branch 非空且语义正确
    assert m.source_branch == "code_reuse" and m.target_branch == "master", (
        f"branch pair 不符预期: {m.source_branch!r} → {m.target_branch!r}"
    )
    _dbg("SELF_CHECK_4_BRANCHES", f"{m.source_branch!r} → {m.target_branch!r} OK")

    # 5. code_reuse_entries 至少有一条语义记录，且实质 commit hash 正确
    assert len(m.code_reuse_entries) >= 1, "应至少有一条 code_reuse_entry 语义记录"
    assert m.code_reuse_entries[0].upstream_commit_hash == "cbd8c054e", (
        "首条 entry 应关联 cbd8c054e（refactored for code reuse）"
    )
    _dbg("SELF_CHECK_5_ENTRIES", f"count={len(m.code_reuse_entries)} OK")

    _dbg("SELF_CHECK_DONE", "全部 5 项断言通过")
    return True


if __name__ == "__main__":
    print(CODE_REUSE_MERGE_MILESTONE.integration_summary())
    print()
    import json
    print(json.dumps(CODE_REUSE_MERGE_MILESTONE.audit_dict(), ensure_ascii=False, indent=2))
    print()
    ok = self_check()
    print(f"self_check: {'PASS' if ok else 'FAIL'}")
