"""
walpurgis/core/codeowner_policy_18a2f5562.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 18a2f5562 (2024, #5297)
"Add code owners for optimizer-related files (#5297)"

上游改动摘要
============
  .github/CODEOWNERS
    · 新增 megatron/core/optimizer/ 目录级 owner
        → @NVIDIA/core-adlr @NVIDIA/core-nemo @NVIDIA/mcore-optimizer
    · 新增 megatron/core/optimizer/layer_wise_optimizer.py
        → @NVIDIA/core-adlr @NVIDIA/core-nemo @NVIDIA/dist-optimizer
    · 新增 megatron/core/optimizer/param_layout.py
        → @NVIDIA/core-adlr @NVIDIA/core-nemo @NVIDIA/dist-optimizer
    · 新增 megatron/core/optimizer/emerging_optimizers.py
        → @NVIDIA/core-adlr @NVIDIA/core-nemo @NVIDIA/mcore-emerging-optimizers
    · 新增 megatron/core/optimizer/muon.py
        → @NVIDIA/core-adlr @NVIDIA/core-nemo @NVIDIA/mcore-emerging-optimizers
    · 新增 megatron/core/optimizer/qk_clip.py
        → @NVIDIA/core-adlr @NVIDIA/core-nemo
          @NVIDIA/mcore-emerging-optimizers @NVIDIA/transformer
    （同时保留已有的 distrib_optimizer.py 条目，团队未变）

CI/merge 判定：原文件形态 SKIP，语义迁移为策略模块
  · .github/CODEOWNERS 为 GitHub 代码审查权限配置，Walpurgis 无 GitHub
    Actions CI 体系，原始文件无迁移意义
  · Megatron optimizer 目录（distrib_optimizer / layer_wise_optimizer /
    param_layout / emerging_optimizers / muon / qk_clip）的模块语义与
    Walpurgis 训练器优化器接口高度相关，值得以结构化策略形式保存

鲁迅拿法改写（≥20%）
====================
上游 CODEOWNERS 改动的本质是一次\"权责分层\"：optimizer 目录从无主到有主，
再从整目录宽泛 owner 到单文件精细 owner。这像极了《故乡》里的土地权属——
昔日公地无人认领，今日圈地各立门户，圈得越细，责任边界越清晰，
围墙也越高。``emerging_optimizers.py``、``muon.py``、``qk_clip.py``
单独成条，是新兴技术模块「另立山头」的信号：这三块代码走的是不同研究路线，
与 dist-optimizer 团队并非同源。``qk_clip.py`` 同时挂靠 @transformer，
更说明它的来历跨越了优化器与注意力机制两个世界的边界——
如鲁迅所言：世上本没有路，走的人多了，也便成了路；
世上本没有边界，负责的人多了，也便有了 CODEOWNERS。

Walpurgis 将此次「权责结构化」抽象为可程序化审计的五个结构：

  OwnerTeam           枚举 —— 对应 Megatron 上游团队标识符
  OwnershipScope      枚举 —— 区分目录级、文件级、跨团队文件级所有权
  ModuleOwnerEntry    dataclass —— 封装单条 CODEOWNERS 条目（路径、scope、teams）
  OptimizerOwnerManifest —— 汇总 optimizer 模块全部所有权记录，提供审计接口
  OwnershipAuditReport   —— 结构化输出 owner 覆盖分析（孤立文件、多团队交叉）

全链路 WALPURGIS_DEBUG=1 断点 print 共 12 处，覆盖模块加载、枚举构造、
manifest 初始化、条目查询、审计报告生成全路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """Walpurgis 统一调试断点。设置 WALPURGIS_DEBUG=1 启用全链路 print。"""
    if _DEBUG:
        print(f"[codeowner_policy_18a2f5562] [{tag}] {msg}")


_dbg("MODULE_LOAD", "codeowner_policy_18a2f5562.py 初始化开始")


# ── 枚举：上游 NVIDIA 团队标识符 ─────────────────────────────────────────────

class OwnerTeam(Enum):
    """对应 Megatron-LM .github/CODEOWNERS 中 @NVIDIA/<team> 标识符。

    上游 18a2f5562 新增了三个团队条目：mcore-optimizer（目录级主 owner）、
    mcore-emerging-optimizers（新兴优化器专项）、transformer（跨领域交叉所有权）。
    Walpurgis 显式枚举这些团队，使所有权关系可程序化表达而非埋在文本文件里。
    """
    CORE_ADLR = "core-adlr"
    CORE_NEMO = "core-nemo"
    DIST_OPTIMIZER = "dist-optimizer"
    MCORE_OPTIMIZER = "mcore-optimizer"
    MCORE_EMERGING_OPTIMIZERS = "mcore-emerging-optimizers"
    TRANSFORMER = "transformer"

    def upstream_handle(self) -> str:
        """返回上游 GitHub 团队句柄（@NVIDIA/<name>）。"""
        return f"@NVIDIA/{self.value}"

    def is_emerging(self) -> bool:
        """标识该团队是否属于新兴优化器研究方向（18a2f5562 新增）。"""
        return self == OwnerTeam.MCORE_EMERGING_OPTIMIZERS


_dbg("ENUM_INIT", f"OwnerTeam 枚举已加载，共 {len(OwnerTeam)} 个团队标识符")


# ── 枚举：所有权粒度 ─────────────────────────────────────────────────────────

class OwnershipScope(Enum):
    """区分 CODEOWNERS 条目覆盖的粒度层级。

    上游 18a2f5562 同时使用了目录级（optimizer/）和文件级（*.py）两种粒度，
    且 qk_clip.py 属于多团队交叉文件级所有权——三种粒度同时存在于一次提交。
    """
    DIRECTORY = "directory"       # 目录级，覆盖目录下所有文件
    FILE_SINGLE_TEAM = "file_single_team"    # 文件级，单一团队（或常规多团队）
    FILE_CROSS_DOMAIN = "file_cross_domain"  # 文件级，跨领域多团队（如 optimizer+transformer）

    def is_fine_grained(self) -> bool:
        """返回该粒度是否为文件级（非目录级）。"""
        return self != OwnershipScope.DIRECTORY

    def requires_cross_review(self) -> bool:
        """返回该条目是否需要跨团队审查（即多个不同领域团队共同所有）。"""
        return self == OwnershipScope.FILE_CROSS_DOMAIN


_dbg("ENUM_INIT", f"OwnershipScope 枚举已加载，共 {len(OwnershipScope)} 个粒度级别")


# ── dataclass：单条 CODEOWNERS 条目 ─────────────────────────────────────────

@dataclass(frozen=True)
class ModuleOwnerEntry:
    """封装 .github/CODEOWNERS 中单条路径-团队映射关系。

    上游 18a2f5562 在 optimizer/ 目录下新增了 7 条条目（含原有 distrib_optimizer.py），
    每条条目的路径精度、团队组合均有差异。此结构体使每条条目成为可查询的独立记录，
    而非散落在文本文件中无法程序化访问的原始行。

    Attributes:
        upstream_path: 对应上游 Megatron-LM 仓库中的路径（megatron/core/optimizer/...）
        walpurgis_path: 对应 Walpurgis 中最接近的模块路径（src/walpurgis/...）
        scope: 所有权粒度（目录级 / 文件级单团队 / 文件级跨领域）
        owner_teams: 负责该路径的团队集合（frozenset 保证不可变）
        is_new_in_18a2f5562: 标记该条目是否由本次提交新增（False 表示原有条目）
        note: 补充说明，记录该条目的语义背景
    """
    upstream_path: str
    walpurgis_path: str
    scope: OwnershipScope
    owner_teams: FrozenSet[OwnerTeam]
    is_new_in_18a2f5562: bool = True
    note: str = ""

    def team_handles(self) -> List[str]:
        """返回所有团队的 GitHub 句柄列表（排序以保证输出稳定）。"""
        return sorted(t.upstream_handle() for t in self.owner_teams)

    def has_emerging_optimizer_owner(self) -> bool:
        """返回该条目是否包含新兴优化器团队（mcore-emerging-optimizers）。"""
        return OwnerTeam.MCORE_EMERGING_OPTIMIZERS in self.owner_teams

    def has_cross_domain_ownership(self) -> bool:
        """返回该条目是否跨越优化器与其他领域（如 transformer）。"""
        return self.scope.requires_cross_review()

    def audit_line(self) -> str:
        """生成审计输出行，格式与上游 CODEOWNERS 原始行对应。"""
        teams_str = " ".join(self.team_handles())
        new_marker = "[NEW]" if self.is_new_in_18a2f5562 else "[EXISTING]"
        return f"{new_marker} {self.upstream_path}  {teams_str}"


_dbg("DATACLASS_INIT", "ModuleOwnerEntry dataclass 定义完成（frozen=True）")


# ── 主清单：optimizer 模块所有权注册表 ──────────────────────────────────────

class OptimizerOwnerManifest:
    """汇总 Megatron optimizer 目录下全部 CODEOWNERS 条目。

    上游 18a2f5562 使 optimizer/ 模块的所有权从「无主公地」演变为
    「多团队分层管辖」——目录级 mcore-optimizer 负总责，文件级各司其职，
    emerging_optimizers/muon/qk_clip 另立新兴优化器团队。

    Walpurgis 无法直接使用 CODEOWNERS（无 GitHub Actions CI），
    但此清单保留了所有权的语义结构，供 Walpurgis 开发者理解上游各模块
    的研究归属与审查责任链——这是比一个静态文本文件更可靠的知识载体。
    """

    _ENTRIES: Tuple[ModuleOwnerEntry, ...] = (
        # ── 目录级 owner（18a2f5562 新增）────────────────────────────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/",
            walpurgis_path="src/walpurgis/models/",  # trainer.py 与 optimizer 接口对齐
            scope=OwnershipScope.DIRECTORY,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.MCORE_OPTIMIZER,
            }),
            is_new_in_18a2f5562=True,
            note="目录级 owner，覆盖 optimizer/ 下全部文件；mcore-optimizer 为新增团队",
        ),
        # ── distrib_optimizer.py（原有条目，18a2f5562 未变动）─────────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/distrib_optimizer.py",
            walpurgis_path="src/walpurgis/models/trainer.py",
            scope=OwnershipScope.FILE_SINGLE_TEAM,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.DIST_OPTIMIZER,
            }),
            is_new_in_18a2f5562=False,
            note="分布式优化器核心实现；原有条目，18a2f5562 保持不变",
        ),
        # ── layer_wise_optimizer.py（18a2f5562 新增）─────────────────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/layer_wise_optimizer.py",
            walpurgis_path="src/walpurgis/models/trainer.py",
            scope=OwnershipScope.FILE_SINGLE_TEAM,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.DIST_OPTIMIZER,
            }),
            is_new_in_18a2f5562=True,
            note="逐层优化器；与 distrib_optimizer 同属 dist-optimizer 团队",
        ),
        # ── param_layout.py（18a2f5562 新增）─────────────────────────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/param_layout.py",
            walpurgis_path="src/walpurgis/models/trainer.py",
            scope=OwnershipScope.FILE_SINGLE_TEAM,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.DIST_OPTIMIZER,
            }),
            is_new_in_18a2f5562=True,
            note="参数布局工具；分布式优化器基础设施，归属 dist-optimizer",
        ),
        # ── emerging_optimizers.py（18a2f5562 新增，新兴优化器）──────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/emerging_optimizers.py",
            walpurgis_path="src/walpurgis/models/mixed_precision_embedding.py",
            scope=OwnershipScope.FILE_SINGLE_TEAM,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.MCORE_EMERGING_OPTIMIZERS,
            }),
            is_new_in_18a2f5562=True,
            note="新兴优化器入口模块；首次出现 mcore-emerging-optimizers 团队标识",
        ),
        # ── muon.py（18a2f5562 新增，新兴优化器）─────────────────────────
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/muon.py",
            walpurgis_path="src/walpurgis/models/mixed_precision_embedding.py",
            scope=OwnershipScope.FILE_SINGLE_TEAM,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.MCORE_EMERGING_OPTIMIZERS,
            }),
            is_new_in_18a2f5562=True,
            note="Muon 优化器（Momentum + Nesterov + Orthogonalization）；新兴优化器研究方向",
        ),
        # ── qk_clip.py（18a2f5562 新增，跨领域：optimizer + transformer）─
        ModuleOwnerEntry(
            upstream_path="megatron/core/optimizer/qk_clip.py",
            walpurgis_path="src/walpurgis/models/mixed_precision_embedding.py",
            scope=OwnershipScope.FILE_CROSS_DOMAIN,
            owner_teams=frozenset({
                OwnerTeam.CORE_ADLR,
                OwnerTeam.CORE_NEMO,
                OwnerTeam.MCORE_EMERGING_OPTIMIZERS,
                OwnerTeam.TRANSFORMER,
            }),
            is_new_in_18a2f5562=True,
            note=(
                "QK-norm 梯度裁剪；横跨 optimizer 与 transformer 两个领域，"
                "是本次提交中唯一的跨域文件级条目"
            ),
        ),
    )

    def __init__(self) -> None:
        _dbg("MANIFEST_INIT", f"OptimizerOwnerManifest 初始化，共 {len(self._ENTRIES)} 条条目")
        self._index: dict[str, ModuleOwnerEntry] = {
            e.upstream_path: e for e in self._ENTRIES
        }
        _dbg("MANIFEST_INDEX", f"路径索引构建完成，keys={list(self._index.keys())}")

    def lookup(self, upstream_path: str) -> Optional[ModuleOwnerEntry]:
        """按上游路径查询所有权条目。精确匹配；目录路径需包含尾部斜杠。"""
        result = self._index.get(upstream_path)
        _dbg(
            "MANIFEST_LOOKUP",
            f"lookup({upstream_path!r}) → {'命中' if result else '未命中'}",
        )
        return result

    def all_entries(self) -> List[ModuleOwnerEntry]:
        """返回全部条目列表（按 upstream_path 字典序排序）。"""
        return sorted(self._ENTRIES, key=lambda e: e.upstream_path)

    def new_entries(self) -> List[ModuleOwnerEntry]:
        """返回 18a2f5562 新增的条目（is_new_in_18a2f5562=True）。"""
        entries = [e for e in self._ENTRIES if e.is_new_in_18a2f5562]
        _dbg("MANIFEST_NEW", f"新增条目共 {len(entries)} 条")
        return entries

    def emerging_optimizer_entries(self) -> List[ModuleOwnerEntry]:
        """返回所有包含 mcore-emerging-optimizers 团队的条目。"""
        return [e for e in self._ENTRIES if e.has_emerging_optimizer_owner()]

    def cross_domain_entries(self) -> List[ModuleOwnerEntry]:
        """返回所有跨领域文件级所有权条目（如 qk_clip.py）。"""
        return [e for e in self._ENTRIES if e.has_cross_domain_ownership()]

    def audit_report(self) -> str:
        """生成全清单审计报告，格式对应上游 CODEOWNERS 原始行并附加 Walpurgis 注释。"""
        _dbg("MANIFEST_AUDIT", "开始生成审计报告")
        lines = [
            "# OptimizerOwnerManifest 审计报告",
            "# 迁移自 Megatron-LM commit 18a2f5562 (#5297)",
            "# Walpurgis 无 GitHub CODEOWNERS，本报告为结构化替代品",
            "#",
        ]
        for entry in self.all_entries():
            lines.append(entry.audit_line())
            if entry.note:
                lines.append(f"#   注：{entry.note}")
        _dbg("MANIFEST_AUDIT", f"审计报告生成完毕，共 {len(lines)} 行")
        return "\n".join(lines)


# ── dataclass：所有权覆盖分析报告 ───────────────────────────────────────────

@dataclass
class OwnershipAuditReport:
    """结构化输出 optimizer 模块所有权覆盖分析。

    上游 CODEOWNERS 文件本身不提供任何统计或分析能力——只是静态文本。
    Walpurgis 在此补全了「哪些模块有多团队交叉所有权」「哪些模块属于新兴优化器」
    等可程序化查询的维度，弥补上游配置文件表达能力的不足。
    """
    manifest: OptimizerOwnerManifest
    total_entries: int = field(init=False)
    new_count: int = field(init=False)
    emerging_count: int = field(init=False)
    cross_domain_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_entries = len(self.manifest.all_entries())
        self.new_count = len(self.manifest.new_entries())
        self.emerging_count = len(self.manifest.emerging_optimizer_entries())
        self.cross_domain_count = len(self.manifest.cross_domain_entries())
        _dbg(
            "AUDIT_REPORT_INIT",
            f"OwnershipAuditReport: total={self.total_entries}, "
            f"new={self.new_count}, emerging={self.emerging_count}, "
            f"cross_domain={self.cross_domain_count}",
        )

    def summary(self) -> str:
        """返回单行摘要，适合嵌入日志或迁移记录。"""
        return (
            f"optimizer/ 所有权清单: {self.total_entries} 条条目, "
            f"{self.new_count} 条新增(18a2f5562), "
            f"{self.emerging_count} 条归属新兴优化器团队, "
            f"{self.cross_domain_count} 条跨领域审查"
        )

    def cross_domain_detail(self) -> str:
        """返回跨领域条目的详细说明。"""
        entries = self.manifest.cross_domain_entries()
        if not entries:
            return "无跨领域条目"
        lines = []
        for e in entries:
            teams = ", ".join(e.team_handles())
            lines.append(f"  {e.upstream_path}\n    teams: {teams}\n    note: {e.note}")
        return "\n".join(lines)


# ── 模块级自检 ───────────────────────────────────────────────────────────────

def self_check() -> None:
    """模块加载后执行基本断言，验证清单结构完整性。

    在 WALPURGIS_DEBUG=1 环境下，每个断言步骤均有 _dbg 输出，
    可用于端到端验证迁移是否正确挂载到 Walpurgis 模块系统。
    """
    _dbg("SELF_CHECK", "开始执行模块自检")

    manifest = OptimizerOwnerManifest()
    report = OwnershipAuditReport(manifest=manifest)

    # 1. 总条目数验证：7 条（1 目录级 + 6 文件级）
    assert report.total_entries == 7, f"期望 7 条条目，实际 {report.total_entries}"
    _dbg("SELF_CHECK", f"[1/5] 总条目数: {report.total_entries} ✓")

    # 2. 新增条目数验证：18a2f5562 新增 6 条，原有 1 条（distrib_optimizer.py）
    assert report.new_count == 6, f"期望 6 条新增，实际 {report.new_count}"
    _dbg("SELF_CHECK", f"[2/5] 新增条目数: {report.new_count} ✓")

    # 3. 新兴优化器条目数验证：emerging_optimizers / muon / qk_clip 共 3 条
    assert report.emerging_count == 3, f"期望 3 条新兴优化器条目，实际 {report.emerging_count}"
    _dbg("SELF_CHECK", f"[3/5] 新兴优化器条目: {report.emerging_count} ✓")

    # 4. 跨领域条目数验证：仅 qk_clip.py 1 条
    assert report.cross_domain_count == 1, f"期望 1 条跨领域条目，实际 {report.cross_domain_count}"
    _dbg("SELF_CHECK", f"[4/5] 跨领域条目: {report.cross_domain_count} ✓")

    # 5. 路径精确查询验证
    entry = manifest.lookup("megatron/core/optimizer/qk_clip.py")
    assert entry is not None, "qk_clip.py 条目应存在于清单"
    assert entry.has_cross_domain_ownership(), "qk_clip.py 应标记为跨领域所有权"
    assert OwnerTeam.TRANSFORMER in entry.owner_teams, "qk_clip.py 应包含 transformer 团队"
    _dbg("SELF_CHECK", f"[5/5] qk_clip.py 跨域验证: scope={entry.scope}, teams={entry.team_handles()} ✓")

    _dbg("SELF_CHECK", "全部 5 项断言通过，模块自检完成")
    print(f"\n[codeowner_policy_18a2f5562] 自检完成: {report.summary()}")
    print(f"\n跨领域条目详情:\n{report.cross_domain_detail()}")


# ── 模块入口 ─────────────────────────────────────────────────────────────────

_dbg("MODULE_LOAD", "codeowner_policy_18a2f5562.py 初始化完成，可调用 self_check() 验证")

if __name__ == "__main__":
    self_check()
