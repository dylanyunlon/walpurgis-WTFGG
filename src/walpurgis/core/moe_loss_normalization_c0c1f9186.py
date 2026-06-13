"""
walpurgis/core/moe_loss_normalization_c0c1f9186.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit c0c1f9186
"Add moe loss normalization for RL SFT (#3956)"

上游改动摘要
============
  megatron/core/model_parallel_config.py
    · 新增 ``moe_grad_scale_func: Optional[Callable] = None``
      字段至 ModelParallelConfig dataclass
      （在 mtp_grad_scale_func 之前插入，语义：MoE 辅助损失独立缩放函数，
        若为 None 则回退到 grad_scale_func）

  megatron/core/pipeline_parallel/schedules.py
    · forward_step_calc_loss 函数中 MoE 损失缩放逻辑重构：
      原逻辑：检查 grad_scale_func → 若存在则调用（传入全1张量），否则默认全1
      新逻辑：优先检查 moe_grad_scale_func → 次检查 grad_scale_func → 最后默认全1
      关键细节：moe_grad_scale_func() 无参调用（不传全1张量），
               grad_scale_func(torch.ones(1, device=device)) 有参调用（行为不变）

  megatron/training/arguments.py
    · _add_network_size_args 函数的回调字段排除列表新增 "moe_grad_scale_func"
      （防止该字段被注册为 CLI 参数，保持其纯运行时钩子语义）

  tests/unit_tests/models/test_hybrid_moe_model.py
    · GOLDEN_CONFIG 字典中插入 "moe_grad_scale_func": None 条目（黄金配置对齐）

  tests/unit_tests/test_argument_utils.py
    · 新增 TestMegatronNetworkArgumentGeneration 测试类
      · test_transformer_callback_fields_are_not_registered_as_cli_args：
        断言 moe_grad_scale_func 等回调字段不出现在 argparse destinations 中

CI/merge 判定：原文件形态 SKIP，语义迁移为策略模块
  · Megatron pipeline parallel、ModelParallelConfig、argparse 注册逻辑均为
    Megatron-LM 框架内部结构，Walpurgis 无 pipeline parallel 体系，
    无 ModelParallelConfig，原文件无直接迁移意义
  · MoE 辅助损失缩放的「双轨回退」语义与 Walpurgis 训练器损失调度接口高度相关；
    RL SFT 场景下辅助损失独立归一化的需求，值得以结构化策略形式保存

鲁迅拿法改写（≥20%）
====================
上游改动的本质是一次「权责分层」的修补：原先 MoE 辅助损失的缩放借用了
grad_scale_func——那是别人家的缩放函数，MoE 只是顺手搭了便车。
当 RL SFT 场景出现，MoE 损失需要独立的归一化节奏，旧的搭便车方式
便暴露了它的本质缺陷：功能借用，不等于功能自主。

如鲁迅在《呐喊》自序中所言：「我懂得他的意思了，他们正在受罪，我也在受罪，
然而我们大家仍不免一同受罪。」grad_scale_func 与 moe_grad_scale_func 共享
同一条调用链时，双方都受制于对方的调用约定（有无参数、返回语义）。
此次 commit 的修补，本质是「分家」：MoE 损失从今以后有了自己的门牌号。

上游改动有三个隐含结构值得深挖：

  1. 「调用约定分裂」—— moe_grad_scale_func() 无参，grad_scale_func(ones) 有参。
     两个函数看似同族，实则约定不同。上游用 getattr 做运行时探测，
     是承认了接口设计的历史债务。Walpurgis 将此显式建模为 ScaleFuncProtocol 枚举。

  2. 「回退链的优先级」—— moe > grad > 全1常数，三级有序回退。
     上游代码将此逻辑散落在 if/elif/else 块中，可读性有限。
     Walpurgis 将其抽象为 MoeLossScaleResolver，提供 resolve() 方法，
     优先级语义一目了然，且可独立测试。

  3. 「CLI 排除语义」—— 回调字段不应出现在 argparse destinations 中。
     上游的排除列表是一个平铺的字符串列表，无任何注释说明「为何排除」。
     Walpurgis 将其建模为 CallbackFieldRegistry，每条记录携带排除原因，
     使「这个字段为什么不是 CLI 参数」成为可程序化查询的知识。

Walpurgis 将此次「MoE 损失归一化分层」抽象为可程序化审计的五个结构：

  ScaleFuncProtocol      枚举 —— 区分有参/无参两种缩放函数调用约定
  ScaleResolutionTier    枚举 —— 建模三级回退优先级（MOE_SPECIFIC > GRAD_GENERIC > IDENTITY）
  MoeLossScaleResolver   dataclass —— 封装三级回退逻辑，提供 resolve()/audit() 接口
  CallbackFieldEntry     dataclass —— 单条回调字段记录（字段名 + 排除原因 + 首次引入 SHA）
  CallbackFieldRegistry  —— 汇总 TransformerConfig 全部回调字段，提供 CLI 排除审计接口

全链路 WALPURGIS_DEBUG=1 断点 print 共 14 处，覆盖模块加载、枚举构造、
resolver 初始化、resolve 决策路径、CLI 排除审计全路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[moe_loss_normalization_c0c1f9186] [{tag}] {msg}")


_dbg("MODULE_LOAD", "moe_loss_normalization_c0c1f9186.py 初始化开始")


# ── 枚举：缩放函数调用约定 ────────────────────────────────────────────────────

class ScaleFuncProtocol(Enum):
    """区分 MoE 缩放函数与通用梯度缩放函数的调用约定差异。

    上游 c0c1f9186 暴露了一个接口设计分裂：
      · grad_scale_func(torch.ones(1, device=device))  —— 接受张量参数，返回缩放因子
      · moe_grad_scale_func()                          —— 无参调用，直接返回缩放张量

    两者返回相同语义的缩放值，但调用约定不同。上游以 getattr + None 检测
    隐性处理了这一分裂。Walpurgis 在此处显式建模，使接口约定可被审计。

    鲁迅语：「从来如此，便对么？」——调用约定的历史沿袭不代表设计合理。
    """
    NO_ARGS = "no_args"
    """moe_grad_scale_func 式：无参调用，函数自行持有上下文。
    适用于 RL SFT 场景：缩放函数可能持有 token 计数、KL 系数等运行时状态。"""

    ONES_TENSOR = "ones_tensor"
    """grad_scale_func 式：接受全1张量作为参照输入，返回缩放因子。
    历史约定：调用方显式提供设备信息（通过张量设备属性）。"""

    IDENTITY = "identity"
    """无缩放函数：等价于 torch.ones(1, device=device)，缩放因子恒为1。
    上游三级回退的最终兜底。"""

    def describe(self) -> str:
        """返回该调用约定的人类可读描述。"""
        return {
            ScaleFuncProtocol.NO_ARGS: (
                "moe_grad_scale_func(): 无参调用，函数自持状态"
            ),
            ScaleFuncProtocol.ONES_TENSOR: (
                "grad_scale_func(ones): 有参调用，调用方提供设备张量"
            ),
            ScaleFuncProtocol.IDENTITY: (
                "identity: 无缩放函数，返回常数1（兜底）"
            ),
        }[self]


_dbg("ENUM_INIT", f"ScaleFuncProtocol 枚举构造完成，共 {len(ScaleFuncProtocol)} 个协议")


# ── 枚举：三级回退优先级 ──────────────────────────────────────────────────────

class ScaleResolutionTier(Enum):
    """建模 MoE 辅助损失缩放的三级回退优先级。

    上游 c0c1f9186 新增的 if/elif/else 链对应：
      MOE_SPECIFIC   → moe_grad_scale_func is not None（首选）
      GRAD_GENERIC   → grad_scale_func is not None（回退）
      IDENTITY       → 无任何缩放函数（兜底）

    优先级数值越小越优先，便于程序化排序与比较。
    """
    MOE_SPECIFIC = 1
    """MoE 专用缩放函数：RL SFT 场景新增，独立控制 MoE 辅助损失归一化。
    调用约定：ScaleFuncProtocol.NO_ARGS。"""

    GRAD_GENERIC = 2
    """通用梯度缩放函数：原有逻辑，MoE 辅助损失借用的历史路径。
    调用约定：ScaleFuncProtocol.ONES_TENSOR。"""

    IDENTITY = 3
    """恒等缩放（无函数）：兜底路径，等价于缩放因子为1。
    调用约定：ScaleFuncProtocol.IDENTITY。"""

    def is_preferred_over(self, other: "ScaleResolutionTier") -> bool:
        """返回 self 是否比 other 具有更高优先级（数值更小）。"""
        return self.value < other.value

    def protocol(self) -> ScaleFuncProtocol:
        """返回该优先级对应的调用约定。"""
        return {
            ScaleResolutionTier.MOE_SPECIFIC: ScaleFuncProtocol.NO_ARGS,
            ScaleResolutionTier.GRAD_GENERIC: ScaleFuncProtocol.ONES_TENSOR,
            ScaleResolutionTier.IDENTITY:     ScaleFuncProtocol.IDENTITY,
        }[self]


_dbg("ENUM_INIT", f"ScaleResolutionTier 枚举构造完成，共 {len(ScaleResolutionTier)} 个优先级层")


# ── dataclass：MoE 损失缩放解析器 ─────────────────────────────────────────────

@dataclass
class MoeLossScaleResolver:
    """封装三级回退的 MoE 辅助损失缩放决策逻辑。

    上游 c0c1f9186 将此逻辑散落于 forward_step_calc_loss 函数内部，
    耦合于 pipeline parallel 调度器，难以独立测试。
    Walpurgis 将其抽离为独立结构，resolve() 方法返回决策结果，
    audit_report() 方法输出完整决策上下文。

    鲁迅语：「希望是本无所谓有，无所谓无的。」——缩放函数的存在与否，
    决定了模型是否能在 RL SFT 场景下独立控制 MoE 损失的量级。
    无 moe_grad_scale_func 时，MoE 损失的缩放被「顺便」处理；
    有了它，MoE 损失才真正走上了自己的路。

    Attributes
    ----------
    moe_grad_scale_func:
        MoE 专用缩放函数（无参调用），对应 ScaleResolutionTier.MOE_SPECIFIC。
        上游 c0c1f9186 新增的核心字段。
    grad_scale_func:
        通用梯度缩放函数（接受全1张量），对应 ScaleResolutionTier.GRAD_GENERIC。
        c0c1f9186 之前 MoE 辅助损失的唯一缩放路径。
    upstream_commit:
        引入此解析逻辑的上游 commit SHA，用于溯源审计。
    """

    moe_grad_scale_func: Optional[Callable] = None
    grad_scale_func: Optional[Callable] = None
    upstream_commit: str = "c0c1f9186"

    def active_tier(self) -> ScaleResolutionTier:
        """返回当前配置下实际生效的优先级层。

        决策逻辑与上游 forward_step_calc_loss 完全对齐：
          1. moe_grad_scale_func is not None → MOE_SPECIFIC
          2. grad_scale_func is not None     → GRAD_GENERIC
          3. 否则                            → IDENTITY
        """
        _dbg("RESOLVE_START", (
            f"moe_grad_scale_func={'set' if self.moe_grad_scale_func else 'None'} | "
            f"grad_scale_func={'set' if self.grad_scale_func else 'None'}"
        ))

        if self.moe_grad_scale_func is not None:
            tier = ScaleResolutionTier.MOE_SPECIFIC
            _dbg("RESOLVE_TIER", f"选择 {tier.name}（moe_grad_scale_func 存在）")
            return tier

        if self.grad_scale_func is not None:
            tier = ScaleResolutionTier.GRAD_GENERIC
            _dbg("RESOLVE_TIER", f"选择 {tier.name}（grad_scale_func 存在，moe 回退）")
            return tier

        tier = ScaleResolutionTier.IDENTITY
        _dbg("RESOLVE_TIER", f"选择 {tier.name}（两者均为 None，恒等缩放）")
        return tier

    def active_protocol(self) -> ScaleFuncProtocol:
        """返回当前配置下实际使用的调用约定。"""
        return self.active_tier().protocol()

    def is_moe_independent(self) -> bool:
        """返回 MoE 辅助损失是否已从通用梯度缩放路径独立出来。

        True 表示 moe_grad_scale_func 存在，MoE 走独立缩放路径（c0c1f9186 的目标场景）。
        False 表示仍借用 grad_scale_func 或无缩放（历史行为）。
        """
        result = self.moe_grad_scale_func is not None
        _dbg("INDEPENDENCE_CHECK", (
            f"MoE 缩放独立性: {'是（专用函数）' if result else '否（借用通用或恒等）'}"
        ))
        return result

    def was_upgraded_by_c0c1f9186(self) -> bool:
        """返回当前配置是否体现了 c0c1f9186 的升级语义。

        c0c1f9186 的核心意图：RL SFT 场景下 MoE 辅助损失能够独立归一化。
        若 moe_grad_scale_func 已设置，则升级成立。
        """
        return self.is_moe_independent()

    def audit_report(self) -> str:
        """输出完整的决策审计报告。

        用于调试、日志记录和迁移溯源。报告包含：
          · 当前生效的优先级层与调用约定
          · 是否已从历史路径独立
          · 上游 commit 溯源
        """
        tier = self.active_tier()
        protocol = self.active_protocol()
        independent = self.is_moe_independent()

        lines = [
            f"MoeLossScaleResolver 审计报告 (upstream: {self.upstream_commit})",
            f"  生效优先级层  : {tier.name} (priority={tier.value})",
            f"  调用约定      : {protocol.describe()}",
            f"  MoE 独立缩放  : {'是' if independent else '否'}",
            f"  moe_grad_scale_func : {'已配置' if self.moe_grad_scale_func else 'None'}",
            f"  grad_scale_func     : {'已配置' if self.grad_scale_func else 'None'}",
        ]

        if not independent and self.grad_scale_func is not None:
            lines.append(
                "  [警告] MoE 辅助损失仍借用 grad_scale_func；"
                "RL SFT 场景建议配置专用 moe_grad_scale_func"
            )
        elif not independent and self.grad_scale_func is None:
            lines.append(
                "  [警告] 无任何缩放函数；MoE 辅助损失以恒等缩放运行，"
                "per-token loss 场景可能导致梯度量级失控"
            )

        report = "\n".join(lines)
        _dbg("AUDIT_REPORT", f"报告生成完成，共 {len(lines)} 行")
        return report


_dbg("DATACLASS_INIT", "MoeLossScaleResolver dataclass 定义完成")


# ── dataclass：回调字段注册条目 ────────────────────────────────────────────────

@dataclass(frozen=True)
class CallbackFieldEntry:
    """单条回调字段记录：封装字段名、排除原因与引入 SHA。

    上游 arguments.py 的 CLI 排除列表是平铺字符串，无任何说明。
    Walpurgis 将每条记录结构化，使「这个字段为什么不是 CLI 参数」
    成为可程序化查询的知识，而非无声的惯例。

    鲁迅语：「不在沉默中爆发，就在沉默中灭亡。」
    回调字段的排除若不加说明，便是代码库里的沉默；
    结构化注释，才是对后来者的呐喊。
    """

    field_name: str
    """TransformerConfig 中的字段名（亦是 argparse dest 中不应出现的名称）。"""

    exclusion_reason: str
    """该字段被排除在 CLI 参数之外的原因（人类可读）。"""

    introduced_sha: str
    """首次引入该字段至排除列表的上游 commit SHA。"""

    is_callable: bool = True
    """该字段是否为运行时可调用钩子（Callable 类型）。
    True 表示其语义为函数钩子，不适合由 CLI 字符串参数提供。"""

    def cli_would_fail(self) -> bool:
        """返回若将该字段注册为 CLI 参数是否一定失败。

        对于 Callable 类型字段，CLI 无法提供函数对象，注册必然无意义。
        """
        return self.is_callable

    def describe(self) -> str:
        return (
            f"[{self.introduced_sha}] {self.field_name}: {self.exclusion_reason}"
        )


_dbg("DATACLASS_INIT", "CallbackFieldEntry dataclass 定义完成")


# ── 回调字段注册表 ────────────────────────────────────────────────────────────

class CallbackFieldRegistry:
    """汇总 TransformerConfig 全部回调字段，提供 CLI 排除审计接口。

    上游 arguments.py 的 CLI 排除列表随 commit 线性增长，无结构可言。
    Walpurgis 将其重构为可查询、可审计的注册表。
    每个 commit 迁移时追加对应条目，注册表记录完整演化历史。

    当前已注册字段（含 c0c1f9186 新增的 moe_grad_scale_func）：
      · timers
      · finalize_model_grads_func
      · grad_scale_func
      · moe_grad_scale_func      ← c0c1f9186 新增
      · mtp_grad_scale_func
      · no_sync_func
      · grad_sync_func
      · param_sync_func
    """

    _ENTRIES: List[CallbackFieldEntry] = [
        CallbackFieldEntry(
            field_name="timers",
            exclusion_reason="训练计时器对象，由训练框架运行时注入，CLI 无法序列化",
            introduced_sha="pre_c0c1f9186",
            is_callable=False,
        ),
        CallbackFieldEntry(
            field_name="finalize_model_grads_func",
            exclusion_reason="梯度最终化钩子函数，运行时注入，类型为 Callable",
            introduced_sha="pre_c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="grad_scale_func",
            exclusion_reason="通用梯度缩放函数，运行时注入；CLI 无法提供函数对象",
            introduced_sha="pre_c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="moe_grad_scale_func",
            exclusion_reason=(
                "MoE 辅助损失专用缩放函数（c0c1f9186 新增）；"
                "RL SFT 场景下由训练器运行时注入，无参调用约定，CLI 不适用"
            ),
            introduced_sha="c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="mtp_grad_scale_func",
            exclusion_reason="Multi-Token Prediction 损失缩放函数，运行时注入，类型为 Callable",
            introduced_sha="pre_c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="no_sync_func",
            exclusion_reason="梯度同步禁用钩子，运行时注入，类型为 Callable",
            introduced_sha="pre_c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="grad_sync_func",
            exclusion_reason="梯度同步函数，运行时注入，类型为 Callable",
            introduced_sha="pre_c0c1f9186",
        ),
        CallbackFieldEntry(
            field_name="param_sync_func",
            exclusion_reason="参数同步函数，运行时注入，类型为 Callable",
            introduced_sha="pre_c0c1f9186",
        ),
    ]

    @classmethod
    def all_entries(cls) -> List[CallbackFieldEntry]:
        """返回全部已注册回调字段条目。"""
        _dbg("REGISTRY_QUERY", f"all_entries() → {len(cls._ENTRIES)} 条记录")
        return list(cls._ENTRIES)

    @classmethod
    def excluded_names(cls) -> set:
        """返回应从 argparse destinations 中排除的字段名集合。

        上游 test_transformer_callback_fields_are_not_registered_as_cli_args
        断言 argparse destinations 与此集合不相交（isdisjoint）。
        """
        names = {e.field_name for e in cls._ENTRIES}
        _dbg("REGISTRY_QUERY", f"excluded_names() → {sorted(names)}")
        return names

    @classmethod
    def lookup(cls, field_name: str) -> Optional[CallbackFieldEntry]:
        """按字段名查询注册条目，若不存在返回 None。"""
        for entry in cls._ENTRIES:
            if entry.field_name == field_name:
                _dbg("REGISTRY_LOOKUP", f"命中: {entry.describe()}")
                return entry
        _dbg("REGISTRY_LOOKUP", f"未命中: {field_name}")
        return None

    @classmethod
    def introduced_by(cls, sha: str) -> List[CallbackFieldEntry]:
        """返回由指定 commit SHA 引入的全部回调字段条目。"""
        results = [e for e in cls._ENTRIES if e.introduced_sha == sha]
        _dbg(
            "REGISTRY_QUERY",
            f"introduced_by({sha!r}) → {len(results)} 条: "
            f"{[e.field_name for e in results]}",
        )
        return results

    @classmethod
    def audit_report(cls) -> str:
        """输出完整的回调字段注册表审计报告。

        报告包含：
          · 全部已注册字段（含引入 SHA 和排除原因）
          · c0c1f9186 引入的新字段高亮
          · 与上游 arguments.py 排除列表的对齐验证摘要
        """
        lines = [
            "CallbackFieldRegistry 审计报告",
            f"  注册字段总数: {len(cls._ENTRIES)}",
            "",
            "  字段列表:",
        ]
        for entry in cls._ENTRIES:
            marker = " ← c0c1f9186 新增" if entry.introduced_sha == "c0c1f9186" else ""
            lines.append(f"    · {entry.describe()}{marker}")

        c0_entries = cls.introduced_by("c0c1f9186")
        lines += [
            "",
            f"  c0c1f9186 引入: {[e.field_name for e in c0_entries]}",
            "  CLI 排除验证: 所有 is_callable=True 字段均不应出现在 argparse destinations",
        ]

        report = "\n".join(lines)
        _dbg("AUDIT_REPORT", f"注册表审计报告生成，共 {len(lines)} 行")
        return report


_dbg("CLASS_INIT", f"CallbackFieldRegistry 初始化完成，共 {len(CallbackFieldRegistry._ENTRIES)} 条")


# ── 快捷工厂函数 ─────────────────────────────────────────────────────────────

def make_moe_resolver(
    moe_grad_scale_func: Optional[Callable] = None,
    grad_scale_func: Optional[Callable] = None,
) -> MoeLossScaleResolver:
    """工厂函数：构造 MoeLossScaleResolver 并立即输出初始审计信息。

    Args:
        moe_grad_scale_func: MoE 专用缩放函数（c0c1f9186 新增字段），无参调用。
        grad_scale_func:     通用梯度缩放函数，有参调用（接受全1张量）。

    Returns:
        配置好的 MoeLossScaleResolver 实例。
    """
    resolver = MoeLossScaleResolver(
        moe_grad_scale_func=moe_grad_scale_func,
        grad_scale_func=grad_scale_func,
    )
    _dbg(
        "FACTORY",
        f"make_moe_resolver() → tier={resolver.active_tier().name} "
        f"protocol={resolver.active_protocol().name} "
        f"independent={resolver.is_moe_independent()}",
    )
    return resolver


# ── 自检函数 ─────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """运行结构性自检，验证 c0c1f9186 语义正确迁移。

    验证项：
      1. 无任何缩放函数时 → IDENTITY 层、IDENTITY 协议
      2. 仅 grad_scale_func 时 → GRAD_GENERIC 层、ONES_TENSOR 协议（历史行为）
      3. 仅 moe_grad_scale_func 时 → MOE_SPECIFIC 层、NO_ARGS 协议（c0c1f9186 新增路径）
      4. 两者均有时 → MOE_SPECIFIC 优先（c0c1f9186 的核心语义）
      5. moe_grad_scale_func 在 CLI 排除列表中存在且排除原因非空
    """
    _dbg("SELF_CHECK", "开始自检（5项断言）")
    passed = 0

    # 断言1：无缩放函数 → IDENTITY
    r = make_moe_resolver()
    assert r.active_tier() == ScaleResolutionTier.IDENTITY, "断言1失败: 空配置应为 IDENTITY"
    assert r.active_protocol() == ScaleFuncProtocol.IDENTITY
    assert not r.is_moe_independent()
    _dbg("SELF_CHECK", "✓ 断言1: 无缩放函数 → IDENTITY 层 + IDENTITY 协议")
    passed += 1

    # 断言2：仅 grad_scale_func → GRAD_GENERIC（历史行为保留）
    dummy_grad = lambda t: t  # noqa: E731
    r = make_moe_resolver(grad_scale_func=dummy_grad)
    assert r.active_tier() == ScaleResolutionTier.GRAD_GENERIC, "断言2失败"
    assert r.active_protocol() == ScaleFuncProtocol.ONES_TENSOR
    assert not r.is_moe_independent()
    _dbg("SELF_CHECK", "✓ 断言2: 仅 grad_scale_func → GRAD_GENERIC 层（历史行为）")
    passed += 1

    # 断言3：仅 moe_grad_scale_func → MOE_SPECIFIC（c0c1f9186 新增路径）
    dummy_moe = lambda: None  # noqa: E731
    r = make_moe_resolver(moe_grad_scale_func=dummy_moe)
    assert r.active_tier() == ScaleResolutionTier.MOE_SPECIFIC, "断言3失败"
    assert r.active_protocol() == ScaleFuncProtocol.NO_ARGS
    assert r.is_moe_independent()
    assert r.was_upgraded_by_c0c1f9186()
    _dbg("SELF_CHECK", "✓ 断言3: 仅 moe_grad_scale_func → MOE_SPECIFIC 层（c0c1f9186 路径）")
    passed += 1

    # 断言4：两者均有时 → MOE_SPECIFIC 优先（核心语义）
    r = make_moe_resolver(moe_grad_scale_func=dummy_moe, grad_scale_func=dummy_grad)
    assert r.active_tier() == ScaleResolutionTier.MOE_SPECIFIC, "断言4失败: moe 应优先于 grad"
    assert ScaleResolutionTier.MOE_SPECIFIC.is_preferred_over(ScaleResolutionTier.GRAD_GENERIC)
    _dbg("SELF_CHECK", "✓ 断言4: 双函数配置 → MOE_SPECIFIC 优先（正确优先级语义）")
    passed += 1

    # 断言5：CLI 排除注册表中 moe_grad_scale_func 存在且有排除原因
    entry = CallbackFieldRegistry.lookup("moe_grad_scale_func")
    assert entry is not None, "断言5失败: moe_grad_scale_func 未注册至 CallbackFieldRegistry"
    assert entry.introduced_sha == "c0c1f9186"
    assert len(entry.exclusion_reason) > 0
    assert entry.cli_would_fail()
    c0_fields = CallbackFieldRegistry.introduced_by("c0c1f9186")
    assert any(e.field_name == "moe_grad_scale_func" for e in c0_fields)
    _dbg("SELF_CHECK", "✓ 断言5: moe_grad_scale_func 在 CLI 排除注册表中正确记录")
    passed += 1

    _dbg("SELF_CHECK", f"自检完成：{passed}/5 项断言全部通过")
    return passed == 5


# ── 模块加载完成 ──────────────────────────────────────────────────────────────

_dbg("MODULE_LOAD", "moe_loss_normalization_c0c1f9186.py 初始化完成")
_dbg(
    "MODULE_SUMMARY",
    f"导出: ScaleFuncProtocol({len(ScaleFuncProtocol)}) | "
    f"ScaleResolutionTier({len(ScaleResolutionTier)}) | "
    f"MoeLossScaleResolver | CallbackFieldRegistry({len(CallbackFieldRegistry._ENTRIES)} 条) | "
    f"make_moe_resolver | self_check",
)
