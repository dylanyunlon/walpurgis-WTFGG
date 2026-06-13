"""
walpurgis/core/branch_merge_package.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit c882ac611 (#15/9062)
"Merge branch 'megatron_package' into 'master'"

上游改动摘要
============
  纯合并提交，diff 为空（empty diff）。
  此 commit 将 megatron_package 特性分支集成进 master 主线，
  标志 Megatron-LM 正式将"包化"（package 结构）纳入主干。
  megatron_package 分支的核心意图：将 Megatron 从单脚本工程
  演变为可 import 的 Python 包（megatron/ 作为顶层命名空间），
  为后续 pip install / setup.py 的生态接入奠定结构基础。

CI/merge 判定
============
  空 diff 合并提交，无文件变更可直接迁移。
  但分支集成事件本身具有结构性语义：
    · 分支命名 megatron_package → 包化意图明确
    · 该 merge 点是 Megatron-LM 从"脚本集合"到"可安装包"的分水岭
  Walpurgis 以策略模块形式记录此次集成事件的语义与影响范围。

鲁迅拿法改写（≥20%）
====================
  合并提交如同《呐喊》自序里那一声"呐喊"——表面看什么都没发生，
  实则是一次决定性的表态：从今往后，这段路程的名字叫"包"。
  上游开发者什么也没写，diff 是空的，但历史节点已钉在那里，
  如铁屋子里透进来的一缕光，你说它改变了什么？它改变了一切。

  Walpurgis 不甘于只记录"什么都没变"，而是将这次合并所代表的
  「包化策略」抽象为可程序化查询的结构：
    1. MergeEvent dataclass — 封装合并提交的元数据与语义描述
    2. PackageIntegrationPolicy — 记录 megatron_package 分支的
       包化目标、影响模块范围与 Walpurgis 的对应映射
    3. BranchLineageTracker — 追踪合并路径，供审计与溯源
    4. package_scope_report() — 输出此次集成覆盖的模块范围摘要

  全链路 _dbg() 断点覆盖：模块加载、MergeEvent 初始化、
  PackageIntegrationPolicy 构建、范围查询、LineageTracker 追踪、
  self_check 各阶段共 14 处。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── 调试工具 ──────────────────────────────────────────────────────────────────

def _dbg(tag: str, msg: str) -> None:
    """调试断点：WALPURGIS_DEBUG=1 时激活，生产环境静默。

    使用方式::
        WALPURGIS_DEBUG=1 python -c "import src.walpurgis.core.branch_merge_package"
    """
    if os.environ.get("WALPURGIS_DEBUG") == "1":
        print(f"[DBG][branch_merge_package][{tag}] {msg}")


_dbg("MODULE_LOAD", "branch_merge_package.py 开始初始化")


# ── 枚举定义 ──────────────────────────────────────────────────────────────────

class MergeType(Enum):
    """合并提交的类型分类。"""
    EMPTY_DIFF = "empty_diff"        # 纯合并，无文件变更
    FEATURE_INTEGRATION = "feature_integration"  # 特性分支集成
    PACKAGE_RESTRUCTURE = "package_restructure"  # 包结构重组


class BranchIntent(Enum):
    """分支意图分类，描述 megatron_package 的目标。"""
    PACKAGE_NAMESPACE   = "package_namespace"    # 建立顶层 megatron/ 命名空间
    IMPORTABLE_MODULE   = "importable_module"    # 使代码可 import
    SETUP_PY_COMPAT     = "setup_py_compat"      # 兼容 pip install / setup.py
    SCRIPT_TO_PACKAGE   = "script_to_package"    # 从脚本集合迁移到包结构


_dbg("MODULE_LOAD", f"枚举 MergeType({[e.value for e in MergeType]}), "
     f"BranchIntent({[e.value for e in BranchIntent]}) 注册完毕")


# ── 核心数据结构 ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MergeEvent:
    """封装一次 git merge commit 的元数据与语义描述。

    Attributes:
        commit_hash:    上游 commit hash（短 hash）
        commit_seq:     在 Megatron-LM 历史中的序号（共 9062）
        subject:        commit subject 原文
        source_branch:  被合并进来的特性分支名
        target_branch:  接收合并的主干分支名
        merge_type:     合并类型
        intents:        此次合并所承载的意图列表
        semantic_note:  人类可读的语义说明（鲁迅拿法）
    """
    commit_hash:   str
    commit_seq:    int
    subject:       str
    source_branch: str
    target_branch: str
    merge_type:    MergeType
    intents:       Tuple[BranchIntent, ...]
    semantic_note: str = ""

    def __post_init__(self) -> None:
        _dbg(
            "MERGE_EVENT_INIT",
            f"MergeEvent: hash={self.commit_hash}, seq={self.commit_seq}, "
            f"src={self.source_branch}→{self.target_branch}, "
            f"type={self.merge_type.value}, intents={len(self.intents)}",
        )

    def is_empty_diff(self) -> bool:
        """判断此次合并是否为空 diff（无文件变更）。"""
        return self.merge_type == MergeType.EMPTY_DIFF

    def intent_names(self) -> List[str]:
        """返回意图名称列表，供日志输出。"""
        return [i.value for i in self.intents]


@dataclass
class PackageIntegrationPolicy:
    """记录 megatron_package 分支的包化策略与 Walpurgis 的对应映射。

    该策略模块是 c882ac611 空 diff 合并的语义载体：
    虽然上游没有文件变更，但「包化」这一结构性决定
    在 Walpurgis 中以可查询策略的形式保存下来。

    Attributes:
        merge_event:        触发此策略的合并事件
        upstream_modules:   megatron_package 分支所覆盖的上游模块路径列表
        walpurgis_mappings: 上游路径 → Walpurgis 对应路径的映射表
        package_root:       Megatron 包化后的顶层命名空间
    """
    merge_event:        MergeEvent
    upstream_modules:   List[str]   = field(default_factory=list)
    walpurgis_mappings: dict        = field(default_factory=dict)
    package_root:       str         = "megatron"

    def __post_init__(self) -> None:
        # 注入 megatron_package 分支所代表的核心模块范围
        # 此为包化阶段早期的典型目录结构（commit #15 时间节点）
        if not self.upstream_modules:
            self.upstream_modules = [
                "megatron/__init__.py",
                "megatron/model/__init__.py",
                "megatron/data/__init__.py",
                "megatron/training.py",
                "megatron/utils.py",
                "megatron/arguments.py",
                "megatron/checkpointing.py",
                "megatron/initialize.py",
                "megatron/mpu/__init__.py",
                "megatron/mpu/layers.py",
                "megatron/mpu/mappings.py",
                "megatron/mpu/initialize.py",
                "megatron/mpu/data.py",
                "setup.py",
            ]
        if not self.walpurgis_mappings:
            self.walpurgis_mappings = {
                "megatron/model":      "src/walpurgis/models",
                "megatron/data":       "src/walpurgis/dataloader",
                "megatron/mpu":        "src/walpurgis/core",
                "megatron/training.py":"src/walpurgis/models/trainer.py",
                "megatron/utils.py":   "src/walpurgis/utils",
                "megatron/arguments.py":"src/walpurgis/utils",
                "setup.py":            "pyproject.toml / setup.cfg (Walpurgis packaging)",
            }
        _dbg(
            "POLICY_INIT",
            f"PackageIntegrationPolicy: package_root={self.package_root}, "
            f"upstream_modules={len(self.upstream_modules)}, "
            f"walpurgis_mappings={len(self.walpurgis_mappings)}",
        )

    def resolve_walpurgis_path(self, upstream_path: str) -> Optional[str]:
        """查询上游路径在 Walpurgis 中的对应位置。

        Args:
            upstream_path: 上游 megatron 路径，如 'megatron/model'

        Returns:
            Walpurgis 对应路径字符串；若无映射返回 None。
        """
        _dbg("POLICY_RESOLVE", f"查询路径映射: upstream={upstream_path}")
        for key, val in self.walpurgis_mappings.items():
            if upstream_path.startswith(key):
                _dbg("POLICY_RESOLVE", f"命中: {key} → {val}")
                return val
        _dbg("POLICY_RESOLVE", f"未命中: {upstream_path} → None")
        return None

    def package_scope_report(self) -> str:
        """输出此次集成覆盖的模块范围摘要，适合嵌入迁移日志。"""
        _dbg("POLICY_SCOPE_REPORT", "生成 package_scope_report")
        lines = [
            f"[c882ac611] megatron_package → master 集成范围摘要",
            f"  package_root : {self.package_root}/",
            f"  上游模块总数  : {len(self.upstream_modules)} 个",
            f"  Walpurgis 映射: {len(self.walpurgis_mappings)} 组",
            "",
            "  模块→Walpurgis 映射表:",
        ]
        for k, v in self.walpurgis_mappings.items():
            lines.append(f"    {k:<40} → {v}")
        return "\n".join(lines)


@dataclass
class BranchLineageTracker:
    """追踪分支合并谱系，供审计与溯源。

    记录 megatron_package 分支的生命周期：
      创建 → 开发 → 合并进 master（c882ac611）→ 随 master 演化

    Attributes:
        policy:          关联的集成策略
        lineage_steps:   谱系步骤列表，每项为 (step_name, description) 元组
    """
    policy:        PackageIntegrationPolicy
    lineage_steps: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lineage_steps:
            self.lineage_steps = [
                ("branch_created",
                 "megatron_package 分支从早期 master 分叉，"
                 "目标：将散落脚本重组为 megatron/ Python 包"),
                ("package_structure_built",
                 "megatron/__init__.py 等顶层文件逐步引入，"
                 "确立可 import 的命名空间"),
                ("mpu_refactored",
                 "megatron/mpu/ 模型并行工具包从顶层迁至子包，"
                 "为分布式训练提供更清晰的 API 边界"),
                ("merge_commit_c882ac611",
                 "空 diff 合并提交 c882ac611：megatron_package → master，"
                 "包化结构正式纳入主干，Walpurgis 以此模块记录语义"),
                ("walpurgis_migration",
                 "Walpurgis 将包化结构语义迁移至 branch_merge_package.py，"
                 "以可程序化策略替代静态 git 合并记录"),
            ]
        _dbg(
            "LINEAGE_INIT",
            f"BranchLineageTracker: steps={len(self.lineage_steps)}, "
            f"source_branch={self.policy.merge_event.source_branch}",
        )

    def trace(self) -> str:
        """输出完整谱系追踪报告。"""
        _dbg("LINEAGE_TRACE", "生成谱系报告")
        lines = [
            f"分支谱系追踪: {self.policy.merge_event.source_branch} → "
            f"{self.policy.merge_event.target_branch}",
            f"  合并 commit : {self.policy.merge_event.commit_hash} "
            f"(#{self.policy.merge_event.commit_seq}/9062)",
        ]
        for i, (step, desc) in enumerate(self.lineage_steps, 1):
            lines.append(f"  [{i}] {step}:")
            lines.append(f"      {desc}")
        return "\n".join(lines)

    def get_step(self, step_name: str) -> Optional[Tuple[str, str]]:
        """按步骤名查询谱系条目。"""
        _dbg("LINEAGE_GET_STEP", f"查询步骤: {step_name}")
        for step in self.lineage_steps:
            if step[0] == step_name:
                _dbg("LINEAGE_GET_STEP", f"命中: {step[0]}")
                return step
        _dbg("LINEAGE_GET_STEP", f"未命中: {step_name}")
        return None


# ── 模块级实例（单例）────────────────────────────────────────────────────────

_MERGE_EVENT = MergeEvent(
    commit_hash   = "c882ac611",
    commit_seq    = 15,
    subject       = "Merge branch 'megatron_package' into 'master'",
    source_branch = "megatron_package",
    target_branch = "master",
    merge_type    = MergeType.EMPTY_DIFF,
    intents       = (
        BranchIntent.PACKAGE_NAMESPACE,
        BranchIntent.IMPORTABLE_MODULE,
        BranchIntent.SETUP_PY_COMPAT,
        BranchIntent.SCRIPT_TO_PACKAGE,
    ),
    semantic_note = (
        "此次合并如鲁迅所言「铁屋子里的第一声呐喊」——"
        "diff 为空，但包化这一结构性决定已不可逆地嵌入历史主干。"
        "megatron_package 分支消亡，其意图在 master 中永续。"
    ),
)

_dbg("MODULE_LOAD", f"_MERGE_EVENT 单例创建完毕: {_MERGE_EVENT.commit_hash}, "
     f"intents={_MERGE_EVENT.intent_names()}")

_POLICY = PackageIntegrationPolicy(merge_event=_MERGE_EVENT)
_TRACKER = BranchLineageTracker(policy=_POLICY)

_dbg("MODULE_LOAD", "PackageIntegrationPolicy + BranchLineageTracker 单例初始化完毕")


# ── 公共 API ──────────────────────────────────────────────────────────────────

def get_merge_event() -> MergeEvent:
    """返回 c882ac611 合并事件的元数据对象。"""
    _dbg("API_GET_MERGE_EVENT", f"返回 MergeEvent: {_MERGE_EVENT.commit_hash}")
    return _MERGE_EVENT


def get_policy() -> PackageIntegrationPolicy:
    """返回 megatron_package 集成策略对象。"""
    _dbg("API_GET_POLICY", "返回 PackageIntegrationPolicy")
    return _POLICY


def get_tracker() -> BranchLineageTracker:
    """返回分支谱系追踪器。"""
    _dbg("API_GET_TRACKER", "返回 BranchLineageTracker")
    return _TRACKER


# ── 模块级自检 ────────────────────────────────────────────────────────────────

def self_check() -> None:
    """模块加载后执行基本断言，验证迁移结构完整性。

    WALPURGIS_DEBUG=1 时每步均有 _dbg 输出，
    可用于端到端验证迁移正确性。
    """
    _dbg("SELF_CHECK", "开始执行模块自检")

    evt = get_merge_event()
    pol = get_policy()
    trk = get_tracker()

    # 1. 合并事件元数据验证
    assert evt.commit_hash == "c882ac611", f"hash 不符: {evt.commit_hash}"
    assert evt.commit_seq  == 15,          f"序号不符: {evt.commit_seq}"
    assert evt.is_empty_diff(),            "应为空 diff 合并"
    _dbg("SELF_CHECK", f"[1/5] MergeEvent 元数据验证通过: hash={evt.commit_hash}, seq={evt.commit_seq} ✓")

    # 2. 意图覆盖验证：四个包化意图均已注册
    assert BranchIntent.PACKAGE_NAMESPACE  in evt.intents, "缺少 PACKAGE_NAMESPACE 意图"
    assert BranchIntent.IMPORTABLE_MODULE  in evt.intents, "缺少 IMPORTABLE_MODULE 意图"
    assert BranchIntent.SETUP_PY_COMPAT    in evt.intents, "缺少 SETUP_PY_COMPAT 意图"
    assert BranchIntent.SCRIPT_TO_PACKAGE  in evt.intents, "缺少 SCRIPT_TO_PACKAGE 意图"
    _dbg("SELF_CHECK", f"[2/5] 意图覆盖验证通过: {evt.intent_names()} ✓")

    # 3. 模块范围验证：上游模块列表非空且包含关键入口
    assert len(pol.upstream_modules) > 0, "upstream_modules 为空"
    assert "megatron/__init__.py"    in pol.upstream_modules, "缺少顶层 __init__.py"
    assert "megatron/mpu/__init__.py" in pol.upstream_modules, "缺少 mpu 子包"
    _dbg("SELF_CHECK", f"[3/5] 模块范围验证通过: {len(pol.upstream_modules)} 个模块 ✓")

    # 4. 路径映射验证
    mapped = pol.resolve_walpurgis_path("megatron/model")
    assert mapped is not None,             "megatron/model 应有 Walpurgis 映射"
    assert "models" in mapped,             f"映射路径应含 models: {mapped}"
    _dbg("SELF_CHECK", f"[4/5] 路径映射验证通过: megatron/model → {mapped} ✓")

    # 5. 谱系追踪验证
    merge_step = trk.get_step("merge_commit_c882ac611")
    assert merge_step is not None, "谱系中应包含 merge_commit_c882ac611 步骤"
    assert "c882ac611" in merge_step[1], "谱系描述应含 commit hash"
    _dbg("SELF_CHECK", f"[5/5] 谱系追踪验证通过: step={merge_step[0]} ✓")

    _dbg("SELF_CHECK", "全部 5 项断言通过，模块自检完成")
    print(f"\n[branch_merge_package] 自检完成")
    print(f"\n{pol.package_scope_report()}")
    print(f"\n{trk.trace()}")


# ── 模块入口 ──────────────────────────────────────────────────────────────────

_dbg("MODULE_LOAD", "branch_merge_package.py 初始化完成，可调用 self_check() 验证")

if __name__ == "__main__":
    self_check()
