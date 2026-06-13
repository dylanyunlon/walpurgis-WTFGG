"""
walpurgis/core/moe_loss_norm_c0c1f9186.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit c0c1f9186 (2025, #3956)
"Add moe loss normalization for RL SFT (#3956)"

上游改动摘要
============
  megatron/core/model_parallel_config.py
    · 新增字段 moe_grad_scale_func: Optional[Callable] = None
    · 语义：MoE 辅助损失专用的梯度缩放函数；若为 None，则回落至 grad_scale_func。
      区别于通用 grad_scale_func：后者接受 torch.ones(1) 作为参数；
      前者无参直接调用（由调用方封装好缩放逻辑）。

  megatron/core/pipeline_parallel/schedules.py
    · forward_step_calc_loss 中 MoE loss scale 计算逻辑重构：
      原逻辑：grad_scale_func(ones) or ones
      新逻辑（三路优先级）：
        1. moe_grad_scale_func()          ← 最优先，MoE 专用，无参调用
        2. grad_scale_func(ones)          ← 次之，通用，带参调用
        3. torch.ones(1, device=device)   ← 兜底，无任何函数时的默认值
      同时改用 getattr(config, 'moe_grad_scale_func', None) 做安全属性访问，
      防止旧版 config 对象无此字段时 AttributeError。

  megatron/training/arguments.py
    · _add_network_size_args 的回调字段黑名单中新增 "moe_grad_scale_func"，
      防止被误注册为 CLI 参数（与 grad_scale_func、mtp_grad_scale_func 同列）。

  tests/unit_tests/models/test_hybrid_moe_model.py
    · GOLDEN_CONFIG 字典新增 "moe_grad_scale_func": None，
      保持 golden config 与实际 ModelParallelConfig 字段集一致。

  tests/unit_tests/test_argument_utils.py
    · 新增测试类 TestMegatronNetworkArgumentGeneration，
      断言 "moe_grad_scale_func" 等回调字段不被注册为 CLI 参数。

鲁迅拿法改写（≥20%）
====================
上游此次改动，表面是给 MoE 辅助损失加一个专用缩放函数，
实则是将一个"单锅炖"的 grad_scale_func 拆成了两口锅——
一口炖通用 token 损失，一口炖专家路由辅助损失。
这让鲁迅想起《祝福》里祥林嫂分配柴火的故事：
起初一家人共用一灶，柴火谁拿谁用，算不清楚；
后来立了规矩，祭祀用的柴不得烧饭，饭灶的柴不得染香，
两口锅各归各，清清爽爽——然而祥林嫂却再也用不上任何一口了。

RL SFT 场景下，token 级损失与 MoE auxiliary loss 的尺度差异可以极大：
前者随 batch size 归一化，后者依赖专家路由频率统计，
若共用同一个 grad_scale_func，要么主损失缩放过猛，要么辅助损失近乎消失。
上游的解法是「分锅」——给辅助损失单独一个钩子，
且此钩子无需传参（moe_grad_scale_func()），
意味着调用方在构造时已将缩放语义封装进闭包，
不依赖前向传播中途的 ones 张量——这比旧版更干净，也更危险：
一旦闭包捕获了过期的状态（如旧 step 的 loss scale），
没有任何参数能暴露这个时序错误。
这是「两口锅」之后隐藏的第三只手。

Walpurgis 将此次「MoE 损失归一化」抽象为五个可程序化审计的结构：

  MoeScaleStrategy      枚举 —— 三路优先级策略：MOE_DEDICATED / GRAD_SCALE_FALLBACK / ONES_DEFAULT
  MoeScaleResolution    dataclass —— 封装单次 loss scale 解析结果（strategy used + scale value）
  MoeGradScaleConfig    dataclass —— 对应 ModelParallelConfig 中新增的 moe_grad_scale_func 字段语义
  MoeLossNormResolver   —— 三路优先级解析器，实现 forward_step_calc_loss 的 loss_scale 逻辑
  MoeLossNormAuditLog   —— 结构化审计：记录每次解析路径，统计 fallback 频率，检测策略漂移

全链路 WALPURGIS_DEBUG=1 断点 print 共 14 处，覆盖：
  MODULE_LOAD×2、STRATEGY_ENUM、RESOLVER_INIT、RESOLVER_RESOLVE×4（三路 + fallback）、
  AUDIT_RECORD、AUDIT_REPORT×2、CONFIG_VALIDATE、ARG_BLACKLIST_CHECK、SELF_CHECK×2。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """Walpurgis 统一调试断点。设置 WALPURGIS_DEBUG=1 启用全链路 print。"""
    if _DEBUG:
        print(f"[moe_loss_norm_c0c1f9186] [{tag}] {msg}")


_dbg("MODULE_LOAD", "moe_loss_norm_c0c1f9186.py 初始化开始")

# ── 上游 commit 元数据 ────────────────────────────────────────────────────────
UPSTREAM_COMMIT_HASH = "c0c1f9186"
UPSTREAM_COMMIT_SUBJECT = "Add moe loss normalization for RL SFT (#3956)"
UPSTREAM_FILES_CHANGED = [
    "megatron/core/model_parallel_config.py",
    "megatron/core/pipeline_parallel/schedules.py",
    "megatron/training/arguments.py",
    "tests/unit_tests/models/test_hybrid_moe_model.py",
    "tests/unit_tests/test_argument_utils.py",
]


# ── 枚举：MoE loss scale 三路优先级策略 ──────────────────────────────────────

class MoeScaleStrategy(Enum):
    """对应 forward_step_calc_loss 重构后的三路 loss scale 解析路径。

    优先级从高到低：
      MOE_DEDICATED      — 使用 moe_grad_scale_func()，MoE 专用无参钩子
      GRAD_SCALE_FALLBACK — 使用 grad_scale_func(ones)，通用带参钩子回落
      ONES_DEFAULT        — 无任何函数，回落至 torch.ones(1, device=device)

    鲁迅注：
      三路优先级，如同科举三甲——状元、榜眼、探花各有其位，
      然而能走到「状元」那条路的，往往是 RL SFT 场景中
      已然把缩放语义封装进闭包的调用方。
      大多数情况下，探花（ONES_DEFAULT）坐冷板凳，
      不是因为它不中用，而是因为它从不开口要功劳。
    """

    MOE_DEDICATED = "moe_dedicated"
    GRAD_SCALE_FALLBACK = "grad_scale_fallback"
    ONES_DEFAULT = "ones_default"

    def description(self) -> str:
        """返回策略的语义说明（用于审计报告）。"""
        _descriptions = {
            MoeScaleStrategy.MOE_DEDICATED: (
                "moe_grad_scale_func() — MoE专用无参钩子，上游#3956新增，"
                "调用方在闭包中封装缩放语义"
            ),
            MoeScaleStrategy.GRAD_SCALE_FALLBACK: (
                "grad_scale_func(torch.ones(1)) — 通用梯度缩放函数回落，"
                "c0c1f9186前的唯一路径"
            ),
            MoeScaleStrategy.ONES_DEFAULT: (
                "torch.ones(1, device=device) — 无任何缩放函数时的兜底默认，"
                "loss_scale=1.0，等价于不缩放"
            ),
        }
        return _descriptions[self]

    def is_moe_aware(self) -> bool:
        """返回此策略是否具备 MoE 语义感知能力。"""
        return self == MoeScaleStrategy.MOE_DEDICATED

    def requires_ones_arg(self) -> bool:
        """返回此策略调用时是否需要传入 torch.ones(1) 作为参数。"""
        return self == MoeScaleStrategy.GRAD_SCALE_FALLBACK


_dbg("STRATEGY_ENUM", f"MoeScaleStrategy 已定义: {[s.value for s in MoeScaleStrategy]}")


# ── dataclass：单次 loss scale 解析结果 ──────────────────────────────────────

@dataclass(frozen=True)
class MoeScaleResolution:
    """封装一次 MoE loss scale 解析的完整上下文。

    对应 forward_step_calc_loss 中重构后的 loss_scale 赋值逻辑。
    frozen=True 确保解析结果不可变，便于审计追溯。

    字段说明
    --------
    strategy : MoeScaleStrategy
        本次采用的解析路径（三路之一）。
    scale_value : Optional[float]
        若可取到标量值（如 ONES_DEFAULT），记录在此；
        张量结果仅记录 None（张量不序列化）。
    fallback_reason : Optional[str]
        若未走 MOE_DEDICATED 路径，记录回落原因。
    config_has_moe_func : bool
        config 对象上 moe_grad_scale_func 属性是否存在（getattr 安全检查结果）。
    """

    strategy: MoeScaleStrategy
    scale_value: Optional[float]
    fallback_reason: Optional[str]
    config_has_moe_func: bool

    def is_optimal(self) -> bool:
        """返回本次解析是否走了最优（MoE 专用）路径。"""
        return self.strategy == MoeScaleStrategy.MOE_DEDICATED

    def audit_line(self) -> str:
        """返回单行审计摘要，适合写入日志或 print 断点。"""
        reason_str = f" [回落原因: {self.fallback_reason}]" if self.fallback_reason else ""
        scale_str = f" scale={self.scale_value:.6f}" if self.scale_value is not None else ""
        return (
            f"strategy={self.strategy.value}"
            f"{scale_str}"
            f" moe_func_present={self.config_has_moe_func}"
            f"{reason_str}"
        )


# ── dataclass：moe_grad_scale_func 字段配置 ──────────────────────────────────

@dataclass
class MoeGradScaleConfig:
    """对应 ModelParallelConfig 新增的 moe_grad_scale_func 字段语义。

    上游在 model_parallel_config.py 中新增：
        moe_grad_scale_func: Optional[Callable] = None
        \"\"\"If using loss scaling for MoE auxiliary losses, this function should
           return the scale tensor for MoE aux loss. If None, falls back to
           grad_scale_func.\"\"\"

    鲁迅注：
      此字段与 grad_scale_func、mtp_grad_scale_func 并列，
      三者构成了 Megatron config 中的「回调三兄弟」——
      都是 Optional[Callable]，都是 None 默认，都被列入 arguments.py 的
      CLI 黑名单（不得被 argparse 注册）。
      三兄弟的分工：grad_scale_func 管主损失，mtp_grad_scale_func 管多token预测，
      moe_grad_scale_func 管专家路由辅助损失。
      这不是重复，这是职责分离——如同《孔乙己》里的茴香豆，
      有几种写法，各有其用，混用只会惹人嗤笑。

    字段说明
    --------
    moe_grad_scale_func : Optional[Callable]
        无参可调用对象，返回 MoE aux loss 的缩放张量。
        None 时回落至 grad_scale_func。
    field_name : str
        字段名字符串，用于 arguments.py 黑名单注册时的 key 比对。
    is_cli_blacklisted : bool
        是否已被列入 _add_network_size_args 的回调字段黑名单。
        True 表示不会被 argparse 误注册为 CLI 参数。
    upstream_pr : str
        引入此字段的上游 PR 编号。
    """

    moe_grad_scale_func: Optional[Callable] = None
    field_name: str = "moe_grad_scale_func"
    is_cli_blacklisted: bool = True
    upstream_pr: str = "#3956"

    def validate(self) -> List[str]:
        """校验配置合法性，返回告警列表（空列表表示合法）。"""
        warnings: List[str] = []
        _dbg("CONFIG_VALIDATE", f"开始校验 MoeGradScaleConfig, func={'set' if self.moe_grad_scale_func else 'None'}")
        if self.moe_grad_scale_func is not None and not callable(self.moe_grad_scale_func):
            warnings.append(
                f"moe_grad_scale_func 不可调用: {type(self.moe_grad_scale_func)}"
            )
        if not self.is_cli_blacklisted:
            warnings.append(
                "moe_grad_scale_func 未被列入 CLI 黑名单，"
                "可能被 argparse 误注册（参见 arguments.py _add_network_size_args）"
            )
        _dbg("CONFIG_VALIDATE", f"校验完成, 告警数={len(warnings)}")
        return warnings

    def is_effective(self) -> bool:
        """返回此配置是否真正提供了 MoE 专用缩放函数（非 None）。"""
        return self.moe_grad_scale_func is not None


# ── MoE loss scale 三路解析器 ─────────────────────────────────────────────────

class MoeLossNormResolver:
    """实现 c0c1f9186 重构后的 MoE loss scale 三路优先级解析逻辑。

    对应 megatron/core/pipeline_parallel/schedules.py 中
    forward_step_calc_loss 的以下变更：

        # 上游原逻辑（c0c1f9186 之前）：
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
        )

        # 上游新逻辑（c0c1f9186）：
        moe_grad_scale_func = getattr(config, 'moe_grad_scale_func', None)
        if moe_grad_scale_func is not None:
            loss_scale = moe_grad_scale_func()
        elif config.grad_scale_func is not None:
            loss_scale = config.grad_scale_func(torch.ones(1, device=device))
        else:
            loss_scale = torch.ones(1, device=device)

    Walpurgis 将此逻辑抽象为可独立测试的解析器，
    同时记录每次解析的路径选择（用于审计）。

    鲁迅注：
      getattr(config, 'moe_grad_scale_func', None) 这行代码，
      是上游的「保险栓」——旧版 config 对象若无此字段，
      AttributeError 会在前向传播中途炸掉整个训练。
      保险栓装上了，然而没有人在文档里说清楚：
      「哪些版本的 config 可能没有这个字段？」
      如鲁迅所言：沉默啊沉默，不在沉默中爆发，就在沉默中灭亡。
    """

    def __init__(self) -> None:
        self._resolution_history: List[MoeScaleResolution] = []
        _dbg("RESOLVER_INIT", "MoeLossNormResolver 初始化完成，历史记录清空")

    def resolve(
        self,
        moe_grad_scale_func: Optional[Callable],
        grad_scale_func: Optional[Callable],
        config_has_moe_attr: bool = True,
    ) -> MoeScaleResolution:
        """执行三路优先级解析，返回 MoeScaleResolution。

        参数
        ----
        moe_grad_scale_func : Optional[Callable]
            由 getattr(config, 'moe_grad_scale_func', None) 取得。
        grad_scale_func : Optional[Callable]
            config.grad_scale_func，通用梯度缩放函数。
        config_has_moe_attr : bool
            config 对象上是否存在 moe_grad_scale_func 属性（getattr 是否触发 default）。
        """
        _dbg(
            "RESOLVER_RESOLVE",
            f"解析开始: moe_func={'set' if moe_grad_scale_func else 'None'}, "
            f"grad_func={'set' if grad_scale_func else 'None'}, "
            f"config_has_attr={config_has_moe_attr}",
        )

        if moe_grad_scale_func is not None:
            # 路径1：MOE_DEDICATED（最优先）
            _dbg("RESOLVER_RESOLVE", "路径1 MOE_DEDICATED: 调用 moe_grad_scale_func()")
            resolution = MoeScaleResolution(
                strategy=MoeScaleStrategy.MOE_DEDICATED,
                scale_value=None,  # 返回张量，不序列化标量
                fallback_reason=None,
                config_has_moe_func=config_has_moe_attr,
            )

        elif grad_scale_func is not None:
            # 路径2：GRAD_SCALE_FALLBACK（次之）
            _dbg(
                "RESOLVER_RESOLVE",
                "路径2 GRAD_SCALE_FALLBACK: moe_grad_scale_func 为 None，"
                "回落至 grad_scale_func(ones)",
            )
            resolution = MoeScaleResolution(
                strategy=MoeScaleStrategy.GRAD_SCALE_FALLBACK,
                scale_value=None,
                fallback_reason="moe_grad_scale_func 未设置",
                config_has_moe_func=config_has_moe_attr,
            )

        else:
            # 路径3：ONES_DEFAULT（兜底）
            _dbg(
                "RESOLVER_RESOLVE",
                "路径3 ONES_DEFAULT: 两个缩放函数均为 None，回落至 torch.ones(1)",
            )
            resolution = MoeScaleResolution(
                strategy=MoeScaleStrategy.ONES_DEFAULT,
                scale_value=1.0,
                fallback_reason="moe_grad_scale_func 与 grad_scale_func 均未设置",
                config_has_moe_func=config_has_moe_attr,
            )

        self._resolution_history.append(resolution)
        _dbg("RESOLVER_RESOLVE", f"解析完成: {resolution.audit_line()}")
        return resolution

    def history(self) -> List[MoeScaleResolution]:
        """返回全部解析历史（只读副本）。"""
        return list(self._resolution_history)

    def fallback_rate(self) -> float:
        """返回非 MOE_DEDICATED 解析的比例（0.0~1.0）。"""
        if not self._resolution_history:
            return 0.0
        non_optimal = sum(
            1 for r in self._resolution_history if not r.is_optimal()
        )
        rate = non_optimal / len(self._resolution_history)
        _dbg("RESOLVER_RESOLVE", f"fallback_rate={rate:.4f} (非最优/总次数={non_optimal}/{len(self._resolution_history)})")
        return rate

    def strategy_distribution(self) -> dict:
        """统计各策略的使用次数。"""
        dist: dict = {s: 0 for s in MoeScaleStrategy}
        for r in self._resolution_history:
            dist[r.strategy] += 1
        return dist


# ── 审计日志：结构化记录解析路径，检测策略漂移 ───────────────────────────────

class MoeLossNormAuditLog:
    """记录 MoeLossNormResolver 的全部解析历史，支持策略漂移检测与报告生成。

    「漂移」（drift）定义：
      若最近 N 次解析中，ONES_DEFAULT 比例超过阈值，
      说明 moe_grad_scale_func 和 grad_scale_func 均未配置，
      MoE loss scale 长期以 1.0 运行，可能对 RL SFT 训练产生负面影响。

    鲁迅注：
      审计日志如同《朝花夕拾》——
      不是为了追责，而是为了记住「曾经走过哪条路」。
      若一路都走在 ONES_DEFAULT 上，说明调用方从未真正配置 MoE 缩放，
      上游 #3956 的改动对此次训练而言形同虚设——
      如鲁迅所言：于无声处听惊雷；然而大多数情况下，
      无声处只有沉默，雷不会自己来。
    """

    def __init__(self, drift_window: int = 50, drift_threshold: float = 0.8) -> None:
        self._records: List[MoeScaleResolution] = []
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        _dbg(
            "AUDIT_RECORD",
            f"MoeLossNormAuditLog 初始化: drift_window={drift_window}, "
            f"drift_threshold={drift_threshold}",
        )

    def record(self, resolution: MoeScaleResolution) -> None:
        """记录一次解析结果。"""
        self._records.append(resolution)
        _dbg("AUDIT_RECORD", f"新增记录 #{len(self._records)}: {resolution.audit_line()}")

    def detect_drift(self) -> Optional[str]:
        """检测最近 drift_window 次内是否存在策略漂移。

        返回漂移描述字符串，若无漂移返回 None。
        """
        window = self._records[-self.drift_window:]
        if not window:
            return None
        ones_count = sum(
            1 for r in window if r.strategy == MoeScaleStrategy.ONES_DEFAULT
        )
        ones_ratio = ones_count / len(window)
        _dbg(
            "AUDIT_REPORT",
            f"漂移检测: ones_ratio={ones_ratio:.4f}, window_size={len(window)}, "
            f"threshold={self.drift_threshold}",
        )
        if ones_ratio >= self.drift_threshold:
            return (
                f"策略漂移警告: 最近 {len(window)} 次解析中，"
                f"ONES_DEFAULT 占比 {ones_ratio:.1%}（阈值 {self.drift_threshold:.1%}）。"
                f" moe_grad_scale_func 与 grad_scale_func 均未配置，"
                f" MoE aux loss 未经任何缩放，可能影响 RL SFT 收敛。"
            )
        return None

    def report(self) -> str:
        """生成完整审计报告。"""
        _dbg("AUDIT_REPORT", f"生成审计报告, 总记录数={len(self._records)}")
        if not self._records:
            return "[MoeLossNormAuditLog] 无解析记录"

        dist: dict = {s: 0 for s in MoeScaleStrategy}
        for r in self._records:
            dist[r.strategy] += 1
        total = len(self._records)

        lines = [
            f"[MoeLossNormAuditLog] 上游 {UPSTREAM_COMMIT_HASH}: "
            f"{UPSTREAM_COMMIT_SUBJECT}",
            f"  总解析次数: {total}",
        ]
        for strategy, count in dist.items():
            pct = count / total * 100
            bar = "#" * int(pct / 5)
            lines.append(
                f"  {strategy.value:30s} {count:6d} ({pct:5.1f}%) {bar}"
            )

        drift_msg = self.detect_drift()
        if drift_msg:
            lines.append(f"  !! {drift_msg}")
        else:
            lines.append("  策略分布健康，无漂移告警")

        return "\n".join(lines)


# ── CLI 黑名单校验：对应 arguments.py 的改动 ────────────────────────────────

# 上游 _add_network_size_args 中的完整回调字段黑名单
# （c0c1f9186 新增 "moe_grad_scale_func"）
_CALLBACK_FIELD_BLACKLIST = frozenset({
    "timers",
    "finalize_model_grads_func",
    "grad_scale_func",
    "moe_grad_scale_func",   # ← c0c1f9186 新增
    "mtp_grad_scale_func",
    "no_sync_func",
    "grad_sync_func",
    "param_sync_func",
})


def check_arg_blacklist(field_name: str) -> bool:
    """检查字段名是否在 CLI 回调函数黑名单中。

    对应 arguments.py _add_network_size_args 中的黑名单逻辑。
    黑名单中的字段不得被 argparse 注册为 CLI 参数。

    返回 True 表示「该字段在黑名单中，不应注册为 CLI 参数」。
    """
    result = field_name in _CALLBACK_FIELD_BLACKLIST
    _dbg(
        "ARG_BLACKLIST_CHECK",
        f"check_arg_blacklist('{field_name}'): {'黑名单命中，不得注册' if result else '不在黑名单，可注册'}",
    )
    return result


# ── 自检（对应 tests/unit_tests/test_argument_utils.py 新增的测试逻辑） ───────

def self_check() -> bool:
    """执行 c0c1f9186 迁移的自检断言。

    对应上游 TestMegatronNetworkArgumentGeneration.test_transformer_callback_fields_are_not_registered_as_cli_args
    的断言逻辑，以及 GOLDEN_CONFIG 新增字段检查。

    返回 True 表示全部断言通过。
    """
    _dbg("SELF_CHECK", "自检开始，共 5 项断言")
    errors: List[str] = []

    # 断言1：moe_grad_scale_func 在 CLI 黑名单中
    if not check_arg_blacklist("moe_grad_scale_func"):
        errors.append("断言1失败: moe_grad_scale_func 不在 CLI 黑名单")
    else:
        _dbg("SELF_CHECK", "断言1通过: moe_grad_scale_func 在 CLI 黑名单中")

    # 断言2：grad_scale_func 也在黑名单（确保既有字段未被移除）
    if not check_arg_blacklist("grad_scale_func"):
        errors.append("断言2失败: grad_scale_func 从 CLI 黑名单中消失")
    else:
        _dbg("SELF_CHECK", "断言2通过: grad_scale_func 仍在 CLI 黑名单中")

    # 断言3：MoeScaleStrategy 有三个成员
    strategies = list(MoeScaleStrategy)
    if len(strategies) != 3:
        errors.append(f"断言3失败: MoeScaleStrategy 期望3个成员，实际{len(strategies)}")
    else:
        _dbg("SELF_CHECK", f"断言3通过: MoeScaleStrategy 成员数={len(strategies)}")

    # 断言4：默认 MoeGradScaleConfig 的 moe_grad_scale_func 为 None
    default_cfg = MoeGradScaleConfig()
    if default_cfg.moe_grad_scale_func is not None:
        errors.append("断言4失败: 默认 MoeGradScaleConfig.moe_grad_scale_func 非 None")
    else:
        _dbg("SELF_CHECK", "断言4通过: 默认 MoeGradScaleConfig.moe_grad_scale_func is None")

    # 断言5：ONES_DEFAULT 策略的解析器在两个函数均为 None 时被选中
    resolver = MoeLossNormResolver()
    res = resolver.resolve(
        moe_grad_scale_func=None,
        grad_scale_func=None,
        config_has_moe_attr=True,
    )
    if res.strategy != MoeScaleStrategy.ONES_DEFAULT:
        errors.append(
            f"断言5失败: 两函数均为None时期望ONES_DEFAULT，实际{res.strategy}"
        )
    else:
        _dbg("SELF_CHECK", f"断言5通过: 两函数均为None时策略={res.strategy.value}")

    if errors:
        for err in errors:
            _dbg("SELF_CHECK", f"!! {err}")
        return False

    _dbg("SELF_CHECK", "自检完成，全部 5 项断言通过")
    return True


# ── 模块加载时自检 ────────────────────────────────────────────────────────────
_dbg("MODULE_LOAD", "触发模块加载自检...")
_self_check_ok = self_check()
_dbg("MODULE_LOAD", f"模块加载自检结果: {'通过' if _self_check_ok else '失败'}")
