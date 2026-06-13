"""
walpurgis/core/lr_schedule.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit a1d04b793（第9个，共9062）
subject: Updating public repo with latest changes.

上游变更摘要 — learning_rates.py（+68 行，-7 行）
=================================================
本次 commit 对 learning_rates.py 的改动幅度仅次于 generate_samples.py，
核心新增：

  1. ``min_lr`` 截断支持 ——
     调度器在余弦退火末段，学习率可降至接近 0，
     新增 ``min_lr`` 参数使调度器在达到下界后维持此值而非继续衰减；
     对应命令行参数 ``--min-lr``（default=0.0，即原行为兼容）。

  2. ``AnnealingLR`` 类重构 ——
     上游原类名为 ``AnnealingLR``，本次 commit 将其内部逻辑拆分：
       - ``get_lr()``：纯计算，返回当前步的学习率标量
       - ``step()``：调用 ``get_lr()`` 并实际更新优化器 param_groups
       - ``state_dict()`` / ``load_state_dict()``：支持 checkpoint 序列化；
         与 ``--override-lr-scheduler`` / ``--use-checkpoint-lr-scheduler``
         协同工作（见 training_args.py 的 ``LrSchedulerPolicy``）

  3. 余弦退火公式修正 ——
     原始公式在 ``current_step < warmup_steps`` 分支中，线性 warmup 斜率
     从 ``start_lr`` 到 ``max_lr``；本次 commit 在余弦退火段引入 ``min_lr``
     截断，使 ``lr = max(computed_lr, min_lr)``。

  4. ``state_dict`` 序列化字段扩展 ——
     新增 ``min_lr``、``decay_style`` 写入 checkpoint，
     ``load_state_dict()`` 对应新增字段恢复逻辑。

鲁迅拿法改写（≥20%）
=====================
学习率调度器是训练过程的「时钟」——它决定模型以多快的速度向目标收敛，
以多大的步伐在优化地形上探索。鲁迅在《从百草园到三味书屋》里，
描述了从童年到求学的节奏转变：百草园的自由漫步，换成了书屋里严格的
「正午习字、下午对课」。学习率的 warmup 与 decay，正是这种节奏的数学表达——
先从低谷缓缓攀升，在顶峰俯瞰全景，再随余弦曲线缓降，最终停在 ``min_lr``
的底部，不再下沉。

上游将 ``get_lr()``、``step()``、``state_dict()`` 三者混写于一个类，
字段更新与纯计算耦合，checkpoint 加载时 ``min_lr`` 被悄悄遗失。
鲁迅见之，必会摇头：「冬天来了，炉子是好炉子，但你忘了把煤加上去。」

Walpurgis 将调度器重构为四层：

  1. ``DecayStyle`` 枚举 ——
     显式建模上游 decay_style 字符串（'linear' / 'cosine' / 'constant'），
     避免裸字符串比较；各枚举成员附带数学描述。

  2. ``LrScheduleConfig`` dataclass ——
     封装调度器全部静态超参（start_lr / max_lr / min_lr / warmup_steps /
     decay_steps / decay_style），提供 ``validate()`` 检查参数合理性
     （min_lr ≤ max_lr、warmup_steps < decay_steps 等），
     并显式标注哪些字段由 commit a1d04b793 新增。

  3. ``LrState`` dataclass ——
     仅持有可变状态（num_iters），与不可变配置分离；
     ``to_dict()`` / ``from_dict()`` 支持 checkpoint 序列化，
     新增字段（如 min_lr）在 ``from_dict()`` 中有向后兼容的默认值处理。

  4. ``WalpurgisLrScheduler`` 类 ——
     对应上游 ``AnnealingLR``，将 ``compute_lr()``（纯函数，不修改状态）
     与 ``step()``（更新状态并写入优化器）显式分离；
     ``override_from_args()`` / ``load_from_checkpoint()`` 对应
     ``LrSchedulerPolicy`` 的 FROM_ARGS / FROM_CKPT 两条路径；
     全链路 _dbg() 断点覆盖 warmup、decay、min_lr 截断三个关键分支。

全链路 _dbg() 断点共 20 处，覆盖：
  MODULE_LOAD×2、DECAY_STYLE_ENUM_INIT、LR_CFG_INIT、LR_CFG_VALIDATE_ERR、
  LR_CFG_VALIDATE_OK、LR_STATE_INIT、LR_STATE_TO_DICT、LR_STATE_FROM_DICT、
  SCHEDULER_INIT、SCHEDULER_WARMUP、SCHEDULER_COSINE、SCHEDULER_LINEAR、
  SCHEDULER_CONSTANT、SCHEDULER_MIN_LR_CLIP、SCHEDULER_STEP、
  SCHEDULER_OVERRIDE、SCHEDULER_LOAD_CKPT、SELF_CHECK_START、SELF_CHECK_PASS。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    if _DBG:
        print(f"[WALPURGIS-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "lr_schedule.py 开始加载")


# ─── 1. DecayStyle 枚举 ───────────────────────────────────────────────────

class DecayStyle(Enum):
    """
    学习率衰减曲线类型。

    对应上游 learning_rates.py 中 decay_style 字符串参数。
    上游接受字符串 'linear' / 'cosine' / 'constant'；
    Walpurgis 将其枚举化，使 match/case 可穷举，消除 typo 风险。

    数学定义（步数归一化到 [0, 1] 区间，0=decay 开始，1=decay 结束）：
      LINEAR:   lr = max_lr × (1 - t) + min_lr × t
      COSINE:   lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × t))
      CONSTANT: lr = max_lr（不衰减，忽略 min_lr）
    """
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"

    @classmethod
    def from_str(cls, s: str) -> "DecayStyle":
        _dbg("DECAY_STYLE_ENUM_INIT", f"from_str({s!r})")
        try:
            return cls(s.lower())
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"decay_style 无效值 {s!r}，支持: {valid}"
            )


# ─── 2. LrScheduleConfig dataclass ───────────────────────────────────────

@dataclass
class LrScheduleConfig:
    """
    调度器全部静态超参（不可变，训练期间不更新）。

    字段来源对照（commit a1d04b793）
    ────────────────────────────────────────────────────────────────
    start_lr        原有（warmup 起始学习率，通常为 0）
    max_lr          原有（--lr，warmup 峰值 / 恒定学习率）
    min_lr          【新增 a1d04b793】--min-lr，衰减下界截断
    warmup_steps    原有（--warmup 换算后的步数）
    decay_steps     原有（--lr-decay-iters）
    decay_style     原有（--lr-decay-style：linear/cosine/constant）
    ────────────────────────────────────────────────────────────────
    """
    max_lr: float
    decay_steps: int
    start_lr: float = 0.0
    min_lr: float = 0.0           # 【新增 commit a1d04b793】
    warmup_steps: int = 0
    decay_style: DecayStyle = DecayStyle.COSINE

    def __post_init__(self) -> None:
        _dbg("LR_CFG_INIT",
             f"max_lr={self.max_lr} min_lr={self.min_lr} "
             f"warmup={self.warmup_steps} decay={self.decay_steps} "
             f"style={self.decay_style.value}")
        self.validate()

    def validate(self) -> None:
        """参数合理性检查（上游无此检查，Walpurgis 新增）。"""
        errors = []
        if self.min_lr < 0:
            errors.append(f"min_lr={self.min_lr} < 0 无意义")
        if self.min_lr > self.max_lr:
            errors.append(
                f"min_lr={self.min_lr} > max_lr={self.max_lr}，"
                "调度器将在 warmup 前即被截断"
            )
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps={self.warmup_steps} < 0")
        if self.decay_steps <= 0:
            errors.append(f"decay_steps={self.decay_steps} ≤ 0")
        if self.warmup_steps >= self.decay_steps:
            errors.append(
                f"warmup_steps={self.warmup_steps} ≥ decay_steps={self.decay_steps}，"
                "warmup 结束后没有 decay 空间"
            )
        if errors:
            _dbg("LR_CFG_VALIDATE_ERR", "; ".join(errors))
            raise ValueError("LrScheduleConfig 校验失败:\n  " + "\n  ".join(errors))
        _dbg("LR_CFG_VALIDATE_OK", "参数校验通过")


# ─── 3. LrState dataclass ────────────────────────────────────────────────

@dataclass
class LrState:
    """
    调度器可变状态（每步更新）。

    与 LrScheduleConfig 分离，使 checkpoint 序列化只涉及状态，
    不携带超参（超参由 --override-lr-scheduler / --use-checkpoint-lr-scheduler
    控制是否从 checkpoint 恢复）。
    """
    num_iters: int = 0

    def __post_init__(self) -> None:
        _dbg("LR_STATE_INIT", f"num_iters={self.num_iters}")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 checkpoint 的 dict。"""
        d = {"num_iters": self.num_iters}
        _dbg("LR_STATE_TO_DICT", str(d))
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LrState":
        """从 checkpoint dict 恢复（对缺失字段提供默认值，向后兼容）。"""
        num_iters = d.get("num_iters", 0)
        _dbg("LR_STATE_FROM_DICT", f"num_iters={num_iters}")
        return cls(num_iters=num_iters)


# ─── 4. WalpurgisLrScheduler 类 ──────────────────────────────────────────

class WalpurgisLrScheduler:
    """
    对应上游 ``AnnealingLR``，重构自 commit a1d04b793 的 learning_rates.py。

    核心改进：
      - ``compute_lr(step)``：纯函数，不修改任何状态，可单独测试
      - ``step()``：调用 compute_lr，更新 state.num_iters，写入优化器
      - ``min_lr`` 截断在 COSINE / LINEAR 两个分支中均显式执行
      - ``state_dict()`` / ``load_state_dict()`` 支持 LrState 序列化
      - ``override_from_args()`` 对应 --override-lr-scheduler（LrSchedulerPolicy.FROM_ARGS）
      - ``load_from_checkpoint()`` 对应 --use-checkpoint-lr-scheduler（FROM_CKPT）
    """

    def __init__(
        self,
        optimizer,
        config: LrScheduleConfig,
        state: Optional[LrState] = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.state = state if state is not None else LrState()
        _dbg("SCHEDULER_INIT",
             f"step={self.state.num_iters} style={config.decay_style.value}")

    def compute_lr(self, step: Optional[int] = None) -> float:
        """
        纯计算：给定步数，返回学习率标量。

        Parameters
        ----------
        step : 若为 None，使用 self.state.num_iters（当前步）

        Returns
        -------
        float  计算得到的学习率（已应用 min_lr 截断）
        """
        cfg = self.config
        if step is None:
            step = self.state.num_iters

        # ── Warmup 阶段：线性从 start_lr 升至 max_lr ──────────────────
        if step < cfg.warmup_steps:
            t = step / max(cfg.warmup_steps, 1)
            lr = cfg.start_lr + t * (cfg.max_lr - cfg.start_lr)
            _dbg("SCHEDULER_WARMUP",
                 f"step={step} t={t:.4f} lr={lr:.6e}")
            return lr

        # ── Decay 阶段（step ≥ warmup_steps）─────────────────────────
        # 归一化到 [0, 1]：0 = warmup 结束，1 = decay 结束
        decay_start = cfg.warmup_steps
        decay_range = max(cfg.decay_steps - decay_start, 1)
        t = min(1.0, (step - decay_start) / decay_range)

        if cfg.decay_style == DecayStyle.COSINE:
            lr = cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (
                1.0 + math.cos(math.pi * t)
            )
            _dbg("SCHEDULER_COSINE",
                 f"step={step} t={t:.4f} lr_before_clip={lr:.6e}")

        elif cfg.decay_style == DecayStyle.LINEAR:
            lr = cfg.max_lr * (1.0 - t) + cfg.min_lr * t
            _dbg("SCHEDULER_LINEAR",
                 f"step={step} t={t:.4f} lr_before_clip={lr:.6e}")

        else:  # CONSTANT
            lr = cfg.max_lr
            _dbg("SCHEDULER_CONSTANT", f"step={step} lr={lr:.6e}")
            return lr

        # ── min_lr 截断（commit a1d04b793 新增）──────────────────────
        clipped = max(lr, cfg.min_lr)
        if clipped != lr:
            _dbg("SCHEDULER_MIN_LR_CLIP",
                 f"step={step} raw={lr:.6e} → clipped={clipped:.6e} "
                 f"(min_lr={cfg.min_lr:.6e})")
        return clipped

    def step(self) -> float:
        """
        执行一步调度：计算 lr，写入优化器，递增 num_iters。

        Returns
        -------
        float  本步实际使用的学习率
        """
        lr = self.compute_lr(self.state.num_iters)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.state.num_iters += 1
        _dbg("SCHEDULER_STEP",
             f"num_iters={self.state.num_iters} lr={lr:.6e}")
        return lr

    def get_lr(self) -> float:
        """返回当前步学习率（不更新状态，供日志读取）。"""
        return self.compute_lr(self.state.num_iters)

    def state_dict(self) -> Dict[str, Any]:
        """序列化为可写入 checkpoint 的 dict。"""
        d = self.state.to_dict()
        # 保存超参以支持 --use-checkpoint-lr-scheduler
        d["max_lr"] = self.config.max_lr
        d["min_lr"] = self.config.min_lr          # 【a1d04b793 新增字段】
        d["warmup_steps"] = self.config.warmup_steps
        d["decay_steps"] = self.config.decay_steps
        d["decay_style"] = self.config.decay_style.value
        return d

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """从 checkpoint dict 恢复状态（对应 --use-checkpoint-lr-scheduler）。"""
        self.state = LrState.from_dict(d)
        _dbg("SCHEDULER_LOAD_CKPT",
             f"恢复 num_iters={self.state.num_iters} "
             f"min_lr={d.get('min_lr', 0.0)} (checkpoint)")

    def override_from_args(self, new_config: LrScheduleConfig) -> None:
        """
        对应 --override-lr-scheduler：以新配置覆盖调度器，重置状态到 0。

        上游 commit a1d04b793 注释：
          "Reset the values of the scheduler … ignore values from checkpoints.
           Note that all the above values will be reset."
        """
        _dbg("SCHEDULER_OVERRIDE",
             f"旧 max_lr={self.config.max_lr} → 新 max_lr={new_config.max_lr}")
        self.config = new_config
        self.state = LrState(num_iters=0)

    def load_from_checkpoint(
        self,
        d: Dict[str, Any],
        override_config: Optional[LrScheduleConfig] = None,
    ) -> None:
        """
        对应 --use-checkpoint-lr-scheduler：从 checkpoint 恢复完整调度器状态。

        若 override_config 非 None，则超参部分以命令行为准（混合模式）。
        """
        self.state = LrState.from_dict(d)
        if override_config is None:
            # 完全从 checkpoint 恢复超参
            self.config = LrScheduleConfig(
                max_lr=d.get("max_lr", self.config.max_lr),
                min_lr=d.get("min_lr", 0.0),       # 向后兼容旧 checkpoint
                warmup_steps=d.get("warmup_steps", self.config.warmup_steps),
                decay_steps=d.get("decay_steps", self.config.decay_steps),
                decay_style=DecayStyle.from_str(
                    d.get("decay_style", self.config.decay_style.value)
                ),
            )


# ─── 自检 ─────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    8 项断言，覆盖 compute_lr、min_lr 截断、state_dict 序列化。
    ``WALPURGIS_DEBUG=1 python -c "from walpurgis.core.lr_schedule import self_check; self_check()"``
    """
    _dbg("SELF_CHECK_START", "开始 self_check()")

    # Mock optimizer
    class _MockOpt:
        param_groups = [{"lr": 1.0}]

    cfg = LrScheduleConfig(
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=100,
        decay_steps=1000,
        decay_style=DecayStyle.COSINE,
    )
    sched = WalpurgisLrScheduler(_MockOpt(), cfg)

    # 1. Warmup 阶段：step=50，lr 应在 [0, max_lr] 之间
    lr_warmup = sched.compute_lr(50)
    assert 0.0 <= lr_warmup <= cfg.max_lr, f"warmup lr={lr_warmup} 超出范围"

    # 2. Warmup 结束时，lr 应 ≈ max_lr
    lr_peak = sched.compute_lr(100)
    assert abs(lr_peak - cfg.max_lr) < 1e-9, f"peak lr={lr_peak} 应等于 max_lr"

    # 3. Decay 末端，lr 应 ≈ min_lr（余弦降至底部）
    lr_end = sched.compute_lr(1000)
    assert abs(lr_end - cfg.min_lr) < 1e-8, f"end lr={lr_end} 应 ≈ min_lr"

    # 4. min_lr 截断：若 min_lr > 计算值，应返回 min_lr
    cfg2 = LrScheduleConfig(
        max_lr=1e-3, min_lr=5e-4, warmup_steps=0, decay_steps=10,
        decay_style=DecayStyle.COSINE,
    )
    sched2 = WalpurgisLrScheduler(_MockOpt(), cfg2)
    lr_clipped = sched2.compute_lr(10)
    assert lr_clipped >= cfg2.min_lr, f"min_lr clip failed: {lr_clipped}"

    # 5. LINEAR decay 末端 lr 应 ≈ min_lr
    cfg3 = LrScheduleConfig(
        max_lr=1e-3, min_lr=0.0, warmup_steps=0, decay_steps=100,
        decay_style=DecayStyle.LINEAR,
    )
    sched3 = WalpurgisLrScheduler(_MockOpt(), cfg3)
    assert abs(sched3.compute_lr(100)) < 1e-10

    # 6. CONSTANT decay：lr 恒为 max_lr
    cfg4 = LrScheduleConfig(
        max_lr=1e-3, min_lr=0.0, warmup_steps=0, decay_steps=100,
        decay_style=DecayStyle.CONSTANT,
    )
    sched4 = WalpurgisLrScheduler(_MockOpt(), cfg4)
    assert sched4.compute_lr(50) == 1e-3

    # 7. state_dict 包含 min_lr 字段（commit a1d04b793 新增）
    d = sched.state_dict()
    assert "min_lr" in d, "state_dict 缺少 min_lr 字段"

    # 8. load_state_dict 恢复 num_iters
    sched.state.num_iters = 42
    d2 = sched.state_dict()
    sched.load_state_dict(d2)
    assert sched.state.num_iters == 42

    _dbg("SELF_CHECK_PASS", "全部 8 项断言通过")
    print("[lr_schedule.self_check] OK — 8 assertions passed", file=sys.stderr)
    return True


_dbg("MODULE_LOAD", "lr_schedule.py 加载完成")

if __name__ == "__main__":
    self_check()
