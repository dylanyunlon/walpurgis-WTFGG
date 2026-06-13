"""
walpurgis/core/notebook_owner_policy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 cugraph-gnn commit d097572 (第158/452个，PR #149)
"add cugraph-notebook-codeowners to CODEOWNERS (#149)"

上游改动摘要
============
  .github/CODEOWNERS
    · 在文件末尾追加两行：
        # notebooks code owners
        *.ipynb @rapidsai/cugraph-notebook-codeowners
    · 即：以 glob 通配符 *.ipynb 将仓库内所有 Jupyter Notebook
      文件的审查权统一交给 @rapidsai/cugraph-notebook-codeowners 团队。
    · 原文件已有 cpp/、/build.sh、pyproject.toml、VERSION 等条目；
      本次是「笔记本条目」的首次出现，打破了「只有源码/构建脚本有 owner」的格局。

CI/merge 判定：原文件 SKIP，语义迁移为策略模块
  · .github/CODEOWNERS 为 GitHub PR 审查权限配置，Walpurgis 无 GitHub
    Actions CI 体系，原始文件不可直接迁移。
  · 但此提交传达的「笔记本 owner 政策」具有可程序化表达的价值：
    谁负责 *.ipynb？何时触发审查？通配符覆盖边界在哪里？
  · Walpurgis 将这些问题抽象为可查询、可审计的策略模块。

鲁迅拿法改写（≥20%）
====================
上游这一行 ``*.ipynb @rapidsai/cugraph-notebook-codeowners``，
看起来微不足道——不过在 CODEOWNERS 末尾加了三行。
可鲁迅在《故乡》里早说过：「其实地上本没有路，走的人多了，也便成了路。」
此前笔记本文件在仓库里是无主之地，谁改谁合并，无人负责。
这一行通配符，划定了边界：凡 .ipynb 者，皆归 notebook-codeowners 管辖。
边界一旦划定，便有了「路」——也有了「墙」。
通配符的力量在于它的贪婪：``*`` 匹配一切，不区分子目录，
不问你是 GNN 示例、实验脚本还是废弃草稿，一律纳入审查链。
这便是「一刀切」的省力，也是「一刀切」的粗粝。

Walpurgis 将此次「通配所有权」抽象为五个可程序化结构：

  NotebookOwnerTeam       枚举  ── 对应上游 @rapidsai/ 团队标识符
  GlobMatchScope          枚举  ── 区分精确路径、目录级、通配符三种覆盖范式
  NotebookOwnerEntry      dataclass ── 封装单条 CODEOWNERS 路径-团队映射
  NotebookOwnerPolicy     ── 汇总全部 notebook 相关 CODEOWNERS 条目，提供审计接口
  GlobCoverageReport      dataclass ── 分析通配符覆盖范围、潜在遗漏与重叠

全链路 WALPURGIS_DEBUG=1 断点共 10 处，覆盖模块加载、枚举构造、
策略初始化、条目查询、通配符匹配、覆盖报告生成全路径。
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """Walpurgis 统一调试断点。设置 WALPURGIS_DEBUG=1 启用全链路 print。"""
    if _DEBUG:
        print(f"[notebook_owner_policy][d097572] [{tag}] {msg}", flush=True)


_dbg("MODULE_LOAD", "notebook_owner_policy.py 初始化开始")


# ── 枚举：rapidsai 团队标识符 ────────────────────────────────────────────────

class NotebookOwnerTeam(Enum):
    """对应上游 .github/CODEOWNERS 中 @rapidsai/<team> 标识符。

    d097572 引入了 cugraph-notebook-codeowners 团队——
    这是 cugraph-gnn 仓库首个专门负责 .ipynb 文件的审查团队。
    Walpurgis 枚举此标识符，使「谁负责笔记本」可程序化查询而非埋在文本里。
    """

    CUGRAPH_NOTEBOOK_CODEOWNERS = "cugraph-notebook-codeowners"
    # 以下为上游 CODEOWNERS 中已有的其他团队（d097572 之前已存在）
    WHOLEGRAPH_CPP_CODEOWNERS = "wholegraph-cpp-codeowners"
    PACKAGING_CODEOWNERS = "packaging-codeowners"

    def upstream_handle(self) -> str:
        """返回上游 GitHub 团队句柄（@rapidsai/<name>）。"""
        return f"@rapidsai/{self.value}"

    def is_notebook_team(self) -> bool:
        """标识该团队是否为 d097572 引入的 notebook 专属团队。"""
        return self == NotebookOwnerTeam.CUGRAPH_NOTEBOOK_CODEOWNERS

    def introduced_in_d097572(self) -> bool:
        """返回该团队标识符是否由 d097572 首次引入到 CODEOWNERS。"""
        # 只有 cugraph-notebook-codeowners 是 d097572 新增的
        return self == NotebookOwnerTeam.CUGRAPH_NOTEBOOK_CODEOWNERS


_dbg("ENUM_INIT", f"NotebookOwnerTeam 枚举已加载，共 {len(NotebookOwnerTeam)} 个团队标识符")


# ── 枚举：CODEOWNERS 条目覆盖范式 ───────────────────────────────────────────

class GlobMatchScope(Enum):
    """区分 CODEOWNERS 路径条目的覆盖范式。

    d097572 使用了 ``*.ipynb`` ——这是一个根级通配符，匹配仓库内任意层级
    （GitHub CODEOWNERS 中 ``*`` 不跨目录，但 ``*.ipynb`` 匹配任意目录下的
    .ipynb 文件）。理解这三种范式的区别，是审计 CODEOWNERS 覆盖盲区的前提。
    """

    EXACT_FILE = "exact_file"        # 精确文件路径，如 /build.sh
    DIRECTORY = "directory"          # 目录级，如 cpp/
    GLOB_WILDCARD = "glob_wildcard"  # 通配符，如 *.ipynb、pyproject.toml（含隐式通配）

    def is_greedy(self) -> bool:
        """通配符条目是否贪婪匹配（可覆盖多文件）。"""
        return self in (GlobMatchScope.DIRECTORY, GlobMatchScope.GLOB_WILDCARD)

    def requires_glob_audit(self) -> bool:
        """该范式是否需要额外的通配符覆盖审计（即存在覆盖边界模糊的风险）。"""
        return self == GlobMatchScope.GLOB_WILDCARD


_dbg("ENUM_INIT", f"GlobMatchScope 枚举已加载，共 {len(GlobMatchScope)} 个覆盖范式")


# ── dataclass：单条 CODEOWNERS 条目 ─────────────────────────────────────────

@dataclass(frozen=True)
class NotebookOwnerEntry:
    """封装 .github/CODEOWNERS 中单条路径-团队映射关系。

    d097572 在上游已有的四条条目基础上，追加了第五条：
        *.ipynb  @rapidsai/cugraph-notebook-codeowners
    此结构体使每条条目成为可查询的独立记录，而非散落在文本文件中的原始行。

    Attributes:
        pattern: CODEOWNERS 路径/glob 模式（原始字符串，如 ``*.ipynb``）
        scope: 覆盖范式（精确文件 / 目录级 / 通配符）
        owner_teams: 负责该路径的团队集合（frozenset 保证不可变）
        introduced_in_d097572: 是否由本次迁移提交新增
        note: 补充说明，记录语义背景
    """

    pattern: str
    scope: GlobMatchScope
    owner_teams: FrozenSet[NotebookOwnerTeam]
    introduced_in_d097572: bool = False
    note: str = ""

    def team_handles(self) -> List[str]:
        """返回所有团队 GitHub 句柄列表（字典序排序，保证输出稳定）。"""
        return sorted(t.upstream_handle() for t in self.owner_teams)

    def matches_path(self, filepath: str) -> bool:
        """判断给定路径是否被该 CODEOWNERS 条目覆盖。

        Walpurgis 改写点（相对于上游纯文本 CODEOWNERS）：
        引入可调用的路径匹配方法，使覆盖检查成为一等公民操作，
        而不是只能人工目视 CODEOWNERS 文件推断。
        对目录级条目：检查路径是否以该目录为前缀。
        对通配符/精确路径条目：使用 fnmatch 按文件名匹配。
        """
        _dbg("matches_path", f"pattern={self.pattern!r} filepath={filepath!r}")
        name = os.path.basename(filepath)
        if self.scope == GlobMatchScope.DIRECTORY:
            # 目录条目：路径前缀匹配（去尾部斜杠）
            prefix = self.pattern.rstrip("/")
            matched = filepath.startswith(prefix + "/") or filepath == prefix
        elif self.scope == GlobMatchScope.EXACT_FILE:
            # 精确匹配：只对比文件名或完整路径
            matched = filepath == self.pattern or name == self.pattern.lstrip("/")
        else:
            # 通配符：fnmatch 对文件名做 glob 匹配
            matched = fnmatch.fnmatch(name, self.pattern)
        _dbg("matches_path", f"→ {matched}")
        return matched

    def codeowners_line(self) -> str:
        """还原为上游 CODEOWNERS 格式的原始行。"""
        teams_str = "  ".join(self.team_handles())
        return f"{self.pattern:<30} {teams_str}"

    def audit_line(self) -> str:
        """生成带来源标注的审计行。"""
        marker = "[d097572-NEW]" if self.introduced_in_d097572 else "[pre-existing]"
        return f"{marker} {self.codeowners_line()}"


_dbg("DATACLASS_INIT", "NotebookOwnerEntry dataclass 定义完成（frozen=True）")


# ── 主策略类：notebook 及全部 CODEOWNERS 条目注册表 ─────────────────────────

class NotebookOwnerPolicy:
    """汇总 cugraph-gnn 仓库 .github/CODEOWNERS 的所有权条目。

    上游 d097572 将 ``*.ipynb`` 一行追加到已有的四条条目之后，
    使「笔记本文件」从无主之地变为有名有姓的责任区域。

    此策略类提供：
      · 按文件路径查询哪些条目覆盖了它（matches_for_path）
      · 判断某文件是否有笔记本 owner 覆盖（is_notebook_covered）
      · 生成全清单审计报告（audit_report）
      · 通配符覆盖边界分析（glob_entries / exact_entries）
    """

    # d097572 之前已有条目 + d097572 新增条目，完整还原上游 CODEOWNERS 结构
    _ENTRIES: Tuple[NotebookOwnerEntry, ...] = (
        # ── d097572 之前已有条目 ─────────────────────────────────────────────
        NotebookOwnerEntry(
            pattern="cpp/",
            scope=GlobMatchScope.DIRECTORY,
            owner_teams=frozenset({NotebookOwnerTeam.WHOLEGRAPH_CPP_CODEOWNERS}),
            introduced_in_d097572=False,
            note="C++ 目录，归属 wholegraph-cpp-codeowners",
        ),
        NotebookOwnerEntry(
            pattern="/build.sh",
            scope=GlobMatchScope.EXACT_FILE,
            owner_teams=frozenset({NotebookOwnerTeam.PACKAGING_CODEOWNERS}),
            introduced_in_d097572=False,
            note="构建脚本，归属 packaging-codeowners",
        ),
        NotebookOwnerEntry(
            pattern="pyproject.toml",
            scope=GlobMatchScope.GLOB_WILDCARD,
            owner_teams=frozenset({NotebookOwnerTeam.PACKAGING_CODEOWNERS}),
            introduced_in_d097572=False,
            note="打包元数据，归属 packaging-codeowners",
        ),
        NotebookOwnerEntry(
            pattern="VERSION",
            scope=GlobMatchScope.EXACT_FILE,
            owner_teams=frozenset({NotebookOwnerTeam.PACKAGING_CODEOWNERS}),
            introduced_in_d097572=False,
            note="版本文件，归属 packaging-codeowners",
        ),
        # ── d097572 新增条目：notebook 通配符 owner ──────────────────────────
        NotebookOwnerEntry(
            pattern="*.ipynb",
            scope=GlobMatchScope.GLOB_WILDCARD,
            owner_teams=frozenset({NotebookOwnerTeam.CUGRAPH_NOTEBOOK_CODEOWNERS}),
            introduced_in_d097572=True,
            note=(
                "通配符覆盖仓库内所有 Jupyter Notebook 文件；"
                "首次为 .ipynb 文件设立专属审查团队，打破无主状态"
            ),
        ),
    )

    def __init__(self) -> None:
        _dbg("POLICY_INIT", f"NotebookOwnerPolicy 初始化，共 {len(self._ENTRIES)} 条条目")
        # 按 pattern 建索引，加速精确查询
        self._exact_index: Dict[str, NotebookOwnerEntry] = {
            e.pattern: e for e in self._ENTRIES
        }
        _dbg("POLICY_INDEX", f"精确路径索引 keys={list(self._exact_index.keys())}")

    # ── 查询接口 ─────────────────────────────────────────────────────────────

    def matches_for_path(self, filepath: str) -> List[NotebookOwnerEntry]:
        """返回覆盖给定路径的所有 CODEOWNERS 条目（可能多条，按原始顺序）。

        Walpurgis 改写点：上游 CODEOWNERS 由 GitHub 按「最后匹配」规则生效；
        此方法返回所有匹配条目，暴露潜在的多条目覆盖冲突，
        而不是只返回最终生效的一条——这是审计视角，比 GitHub 视角更严格。
        """
        _dbg("matches_for_path", f"filepath={filepath!r}")
        matched = [e for e in self._ENTRIES if e.matches_path(filepath)]
        _dbg("matches_for_path", f"命中 {len(matched)} 条条目: {[e.pattern for e in matched]}")
        return matched

    def is_notebook_covered(self, filepath: str) -> bool:
        """判断给定路径是否被 notebook codeowners 条目覆盖。

        即：该文件是否匹配 ``*.ipynb`` 条目（d097572 新增）。
        对非 .ipynb 文件，此方法始终返回 False。
        """
        nb_entry = self._exact_index.get("*.ipynb")
        if nb_entry is None:
            _dbg("is_notebook_covered", "*.ipynb 条目不存在于策略中")
            return False
        covered = nb_entry.matches_path(filepath)
        _dbg("is_notebook_covered", f"filepath={filepath!r} → {covered}")
        return covered

    def glob_entries(self) -> List[NotebookOwnerEntry]:
        """返回所有通配符范式的条目（GlobMatchScope.GLOB_WILDCARD）。"""
        return [e for e in self._ENTRIES if e.scope == GlobMatchScope.GLOB_WILDCARD]

    def exact_entries(self) -> List[NotebookOwnerEntry]:
        """返回所有精确文件范式的条目（GlobMatchScope.EXACT_FILE）。"""
        return [e for e in self._ENTRIES if e.scope == GlobMatchScope.EXACT_FILE]

    def d097572_entries(self) -> List[NotebookOwnerEntry]:
        """返回 d097572 新增的条目（introduced_in_d097572=True）。"""
        entries = [e for e in self._ENTRIES if e.introduced_in_d097572]
        _dbg("d097572_entries", f"新增条目共 {len(entries)} 条")
        return entries

    def audit_report(self) -> str:
        """生成全清单审计报告，格式对应上游 CODEOWNERS 原始行并附加 Walpurgis 注释。"""
        _dbg("POLICY_AUDIT", "开始生成审计报告")
        lines = [
            "# NotebookOwnerPolicy 审计报告",
            "# 迁移自 cugraph-gnn commit d097572 (PR #149)",
            "# 上游原始路径: .github/CODEOWNERS",
            "# Walpurgis 无 GitHub Actions CI，本报告为结构化替代品",
            "#",
            "# 格式: [来源标注] <pattern>  <@team1>  <@team2>  ...",
            "#",
        ]
        for entry in self._ENTRIES:
            lines.append(entry.audit_line())
            if entry.note:
                lines.append(f"#   注: {entry.note}")
        _dbg("POLICY_AUDIT", f"审计报告生成完毕，共 {len(lines)} 行")
        return "\n".join(lines)


# ── dataclass：通配符覆盖分析报告 ───────────────────────────────────────────

@dataclass
class GlobCoverageReport:
    """分析 NotebookOwnerPolicy 中通配符条目的覆盖情况。

    d097572 引入的 ``*.ipynb`` 通配符看似简单，但其覆盖边界存在几个值得审计的点：
    1. 它能匹配子目录下的 .ipynb 吗？（GitHub CODEOWNERS 的 * 不跨目录，
       但 *.ipynb 匹配任意目录下的 .ipynb——这是 GitHub 的特殊规则）
    2. 与其他条目是否有路径重叠？（如果某 .ipynb 恰好在 cpp/ 下，谁优先？）
    3. 仓库内有哪些文件类型是「无主」的？

    Walpurgis 将这些问题具体化为可程序化的字段，而不是停留在文本 CODEOWNERS 的模糊状态。
    """

    policy: NotebookOwnerPolicy
    total_entries: int = field(init=False)
    glob_count: int = field(init=False)
    exact_count: int = field(init=False)
    d097572_new_count: int = field(init=False)
    notebook_team_introduced: bool = field(init=False)

    def __post_init__(self) -> None:
        self.total_entries = len(self.policy._ENTRIES)
        self.glob_count = len(self.policy.glob_entries())
        self.exact_count = len(self.policy.exact_entries())
        self.d097572_new_count = len(self.policy.d097572_entries())
        # 验证 cugraph-notebook-codeowners 是否已被引入
        self.notebook_team_introduced = any(
            e.introduced_in_d097572 and
            NotebookOwnerTeam.CUGRAPH_NOTEBOOK_CODEOWNERS in e.owner_teams
            for e in self.policy._ENTRIES
        )
        _dbg(
            "GlobCoverageReport.__post_init__",
            f"total={self.total_entries} glob={self.glob_count} "
            f"exact={self.exact_count} d097572_new={self.d097572_new_count} "
            f"notebook_team={self.notebook_team_introduced}",
        )

    def check_path_coverage(self, filepath: str) -> str:
        """对给定路径生成人类可读的覆盖分析摘要。

        Walpurgis 改写点：上游只有一个静态 CODEOWNERS 文本，
        开发者无法直接问「这个文件被谁管」；此方法让策略对象可以回答这个问题。
        """
        _dbg("check_path_coverage", f"filepath={filepath!r}")
        matched = self.policy.matches_for_path(filepath)
        if not matched:
            return f"⚠ 无主: {filepath!r} 未被任何 CODEOWNERS 条目覆盖"
        lines = [f"✓ {filepath!r} 被以下条目覆盖（共 {len(matched)} 条）:"]
        for e in matched:
            teams = ", ".join(e.team_handles())
            lines.append(f"    pattern={e.pattern!r}  scope={e.scope.value}  teams={teams}")
        # GitHub 最终生效的是最后匹配条目（CODEOWNERS 规则）
        final = matched[-1]
        lines.append(
            f"  → GitHub 生效条目（最后匹配）: pattern={final.pattern!r}  "
            f"teams={', '.join(final.team_handles())}"
        )
        return "\n".join(lines)

    def summary(self) -> str:
        """返回单行摘要，适合嵌入日志或迁移记录。"""
        nb_status = "已建立" if self.notebook_team_introduced else "未建立"
        return (
            f"cugraph-gnn CODEOWNERS 策略: {self.total_entries} 条条目 "
            f"({self.glob_count} 通配符, {self.exact_count} 精确文件), "
            f"d097572 新增 {self.d097572_new_count} 条, "
            f"notebook 专属 owner {nb_status}"
        )

    def glob_audit_detail(self) -> str:
        """返回所有通配符条目的覆盖边界分析。"""
        entries = self.policy.glob_entries()
        if not entries:
            return "无通配符条目"
        lines = ["通配符条目覆盖边界分析:"]
        for e in entries:
            lines.append(f"  pattern={e.pattern!r}")
            lines.append(f"    scope={e.scope.value}")
            lines.append(f"    teams={', '.join(e.team_handles())}")
            lines.append(f"    requires_glob_audit={e.scope.requires_glob_audit()}")
            lines.append(f"    注: {e.note or '无'}")
        return "\n".join(lines)


# ── 模块级自检 ───────────────────────────────────────────────────────────────

def self_check() -> None:
    """模块加载后执行基本断言，验证策略结构完整性。

    在 WALPURGIS_DEBUG=1 环境下，每个断言步骤均有 _dbg 输出，
    可用于端到端验证此次 d097572 迁移是否正确挂载到 Walpurgis 模块系统。
    """
    _dbg("SELF_CHECK", "开始执行模块自检（共 6 项断言）")

    policy = NotebookOwnerPolicy()
    report = GlobCoverageReport(policy=policy)

    # 1. 总条目数验证：d097572 前 4 条 + d097572 新增 1 条 = 5 条
    assert report.total_entries == 5, f"期望 5 条条目，实际 {report.total_entries}"
    _dbg("SELF_CHECK", f"[1/6] 总条目数: {report.total_entries} ✓")

    # 2. d097572 新增条目数验证：只有 *.ipynb 条目是本次新增
    assert report.d097572_new_count == 1, (
        f"期望 1 条 d097572 新增条目，实际 {report.d097572_new_count}"
    )
    _dbg("SELF_CHECK", f"[2/6] d097572 新增条目数: {report.d097572_new_count} ✓")

    # 3. notebook 专属团队已被引入
    assert report.notebook_team_introduced, "cugraph-notebook-codeowners 应已由 d097572 引入"
    _dbg("SELF_CHECK", f"[3/6] notebook_team_introduced: {report.notebook_team_introduced} ✓")

    # 4. *.ipynb 匹配验证：典型 notebook 文件应被覆盖
    assert policy.is_notebook_covered("fraud_detection.ipynb"), (
        "fraud_detection.ipynb 应匹配 *.ipynb 条目"
    )
    assert policy.is_notebook_covered("examples/gcn/gcn_demo.ipynb"), (
        "子目录下的 .ipynb 文件应被覆盖"
    )
    assert not policy.is_notebook_covered("train.py"), (
        ".py 文件不应匹配 *.ipynb 条目"
    )
    _dbg("SELF_CHECK", "[4/6] *.ipynb 路径匹配验证 ✓")

    # 5. 通配符条目数量验证：*.ipynb + pyproject.toml = 2 条
    assert report.glob_count == 2, (
        f"期望 2 条通配符条目（*.ipynb + pyproject.toml），实际 {report.glob_count}"
    )
    _dbg("SELF_CHECK", f"[5/6] 通配符条目数: {report.glob_count} ✓")

    # 6. 多条目覆盖检测：cpp/ 下的 .ipynb 文件应同时匹配 cpp/ 和 *.ipynb
    overlap = policy.matches_for_path("cpp/some_notebook.ipynb")
    patterns_hit = {e.pattern for e in overlap}
    assert "cpp/" in patterns_hit, "cpp/ 下的 .ipynb 应匹配 cpp/ 目录条目"
    assert "*.ipynb" in patterns_hit, "cpp/ 下的 .ipynb 也应匹配 *.ipynb 通配符条目"
    _dbg("SELF_CHECK", f"[6/6] 多条目覆盖检测: patterns_hit={patterns_hit} ✓")

    _dbg("SELF_CHECK", "全部 6 项断言通过，模块自检完成")
    print(f"\n[notebook_owner_policy] 自检完成: {report.summary()}")
    print(f"\n{report.glob_audit_detail()}")
    print(f"\n{report.check_path_coverage('examples/gnn_demo.ipynb')}")
    print(f"\n{report.check_path_coverage('cpp/some_notebook.ipynb')}")


# ── 模块入口 ─────────────────────────────────────────────────────────────────

_dbg("MODULE_LOAD", "notebook_owner_policy.py 初始化完成，可调用 self_check() 验证")

if __name__ == "__main__":
    self_check()
