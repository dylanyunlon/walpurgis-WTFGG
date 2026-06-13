"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: learning_rates.py

鲁迅拿法改写（≥20%）：
  上游 learning_rates.py 的 AnnealingLR 是一个静默机器：
  每次 step() 悄悄改变学习率，调用者不知从哪儿来，不知到哪儿去。
  warmup 段、cosine 段、linear 段三个逻辑混在一个 if-elif 链里，
  没有状态名，没有阶段边界日志。
  新增的 min_lr 裁断逻辑同样是一行 max()，
  无人记录"今日被裁至下界多少次"。
  鲁迅见过这种默默运转的工厂机器：出了差错也无人知晓，
  直到损失曲线崩塌才发现 lr 已在 0 附近趴了三千步。

  Walpurgis 改写：
  1. 引入 LRPhase 枚举，显式建模 WARMUP / ANNEALING / FLOOR 三阶段
  2. 每次阶段切换记录断点（_dbg）
  3. min_lr 裁断次数计数，超过阈值发出警告
  4. 新增 state_dict / load_state_dict，与 PyTorch scheduler 接口对齐
  5. 新增 override_from_args / restore_from_checkpoint 两个控制路径，
     对应上游新增的 --override-lr-scheduler / --use-checkpoint-lr-scheduler

迁移位置: src/walpurgis/scheduler/lr_scheduler.py
"""

import os
import sys
import math
from enum import Enum
from typing import Optional, Dict, Any

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：学习率调度器关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    print(f"[_dbg:lr_scheduler:{tag}] {msg}", file=sys.stderr, flush=True)


# ── 阶段枚举（Walpurgis特有：原上游无显式阶段概念） ──────────────────────────
class LRPhase(Enum):
    WARMUP = "warmup"       # 线性升温
    ANNEALING = "annealing" # cosine/linear 衰减
    FLOOR = "floor"         # 已触及 min_lr 下界


# ── 主调度器 ──────────────────────────────────────────────────────────────────
class AnnealingLR:
    """
    Megatron-LM 风格的学习率调度器，Walpurgis 改写版。

    原上游接口保持兼容（step() 返回 float），
    新增: phase 属性、state_dict、load_state_dict、
          override_from_args、restore_from_checkpoint。

    参数:
        optimizer       : PyTorch optimizer（持有 param_groups）
        start_lr        : 起始/目标学习率（warmup 结束后的峰值）
        warmup_iter     : warmup 步数（0 = 无 warmup）
        total_iters     : 总训练步数
        decay_style     : 'cosine' | 'linear' | 'constant'
        last_iter       : 从哪一步恢复（默认 -1 = 全新训练）
        min_lr          : 学习率下界（新增，来自 --min-lr）
        use_checkpoint_lr_scheduler : 若 True，load_state_dict 时完全覆盖所有字段
        override_lr_scheduler       : 若 True，load_state_dict 时忽略 checkpoint 值
    """

    def __init__(
        self,
        optimizer,
        start_lr: float,
        warmup_iter: int,
        total_iters: int,
        decay_style: str = "cosine",
        last_iter: int = -1,
        min_lr: float = 0.0,                       # 新增 a1d04b793
        use_checkpoint_lr_scheduler: bool = False,  # 新增 a1d04b793
        override_lr_scheduler: bool = False,        # 新增 a1d04b793
    ) -> None:
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.total_iters = total_iters
        self.decay_style = decay_style
        self.min_lr = min_lr
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        self.override_lr_scheduler = override_lr_scheduler

        # Walpurgis: 互斥检查（与 megatron_args_a1d04b793 保持一致）
        if use_checkpoint_lr_scheduler and override_lr_scheduler:
            raise ValueError(
                "use_checkpoint_lr_scheduler 与 override_lr_scheduler 互斥"
            )

        self._num_iters = last_iter
        self._phase = LRPhase.WARMUP if warmup_iter > 0 else LRPhase.ANNEALING
        self._floor_count = 0  # Walpurgis: 记录触底次数

        self.step(1)  # 初始化 optimizer param_groups

        _dbg(
            "SCHEDULER_INIT",
            f"start_lr={start_lr} warmup={warmup_iter} total={total_iters} "
            f"decay={decay_style} min_lr={min_lr} phase={self._phase.value}",
        )

    # ── 核心计算 ──────────────────────────────────────────────────────────────
    def get_lr(self) -> float:
        """返回当前步的学习率（已含 min_lr 裁断）。"""
        # 区分三个阶段
        if self.warmup_iter > 0 and self._num_iters <= self.warmup_iter:
            # WARMUP: 线性升温
            prev_phase = self._phase
            self._phase = LRPhase.WARMUP
            lr = self.start_lr * self._num_iters / self.warmup_iter
            if prev_phase != LRPhase.WARMUP:
                _dbg("PHASE_TRANSITION", f"-> {self._phase.value} iter={self._num_iters}")
        elif self._num_iters <= self.total_iters:
            # ANNEALING: 衰减段
            prev_phase = self._phase
            self._phase = LRPhase.ANNEALING
            if prev_phase != LRPhase.ANNEALING:
                _dbg("PHASE_TRANSITION", f"-> {self._phase.value} iter={self._num_iters}")

            # 归一化进度 (0 → 1)
            decay_iters = max(self.total_iters - self.warmup_iter, 1)
            progress = (self._num_iters - self.warmup_iter) / decay_iters

            if self.decay_style == "cosine":
                lr = 0.5 * self.start_lr * (1.0 + math.cos(math.pi * progress))
            elif self.decay_style == "linear":
                lr = self.start_lr * (1.0 - progress)
            else:
                # constant
                lr = self.start_lr
        else:
            # 超出 total_iters → 保持 min_lr
            lr = self.min_lr
            self._phase = LRPhase.FLOOR

        # min_lr 裁断（Walpurgis: 计数并在超阈值时警告）
        if lr < self.min_lr:
            self._floor_count += 1
            lr = self.min_lr
            self._phase = LRPhase.FLOOR
            if self._floor_count == 1:
                _dbg("FLOOR_HIT", f"lr clipped to min_lr={self.min_lr} at iter={self._num_iters}")
            elif self._floor_count % 500 == 0:
                _dbg(
                    "FLOOR_SUSTAINED",
                    f"lr at floor for {self._floor_count} steps "
                    f"(iter={self._num_iters}) — 检查是否 total_iters 设置过小",
                )

        return lr

    def step(self, increment: int = 1) -> None:
        """推进 `increment` 步并更新 optimizer lr。"""
        self._num_iters += increment
        new_lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        _dbg("STEP", f"iter={self._num_iters} lr={new_lr:.6e} phase={self._phase.value}")

    @property
    def phase(self) -> LRPhase:
        return self._phase

    @property
    def num_iters(self) -> int:
        return self._num_iters

    # ── 状态持久化（Walpurgis新增，上游无） ──────────────────────────────────
    def state_dict(self) -> Dict[str, Any]:
        """与 PyTorch scheduler 接口对齐，支持 checkpoint 保存。"""
        return {
            "num_iters": self._num_iters,
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "total_iters": self.total_iters,
            "decay_style": self.decay_style,
            "min_lr": self.min_lr,
            "phase": self._phase.value,
            "floor_count": self._floor_count,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """
        从 checkpoint 恢复调度器状态。
        行为由构造时标志决定：
          - override_lr_scheduler=True  → 完全忽略 sd，保持 CLI 参数
          - use_checkpoint_lr_scheduler=True → 完全以 sd 为准
          - 两者皆 False → 恢复 num_iters，其余保持 CLI 参数（默认）
        """
        if self.override_lr_scheduler:
            _dbg("LOAD_STATE", "override_lr_scheduler=True → 忽略 checkpoint scheduler 状态")
            return

        if self.use_checkpoint_lr_scheduler:
            self._num_iters = sd["num_iters"]
            self.start_lr = sd["start_lr"]
            self.warmup_iter = sd["warmup_iter"]
            self.total_iters = sd["total_iters"]
            self.decay_style = sd["decay_style"]
            self.min_lr = sd.get("min_lr", 0.0)
            self._phase = LRPhase(sd.get("phase", LRPhase.WARMUP.value))
            self._floor_count = sd.get("floor_count", 0)
            _dbg("LOAD_STATE", f"use_checkpoint=True → 完全恢复: iter={self._num_iters}")
        else:
            # 默认: 只恢复进度，其余 CLI 覆盖
            self._num_iters = sd["num_iters"]
            _dbg("LOAD_STATE", f"default → 恢复 iter={self._num_iters}，其余保持 CLI")

    # ── 控制路径：对应上游两个新 flag ─────────────────────────────────────────
    def override_from_args(
        self,
        start_lr: float,
        warmup_iter: int,
        total_iters: int,
        decay_style: str,
        min_lr: float = 0.0,
    ) -> None:
        """
        对应 --override-lr-scheduler：
        重置所有调度参数为 CLI 值，清零计步器和触底计数。
        Walpurgis: 上游只有这个标志，没有实现；此处补全。
        """
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.total_iters = total_iters
        self.decay_style = decay_style
        self.min_lr = min_lr
        self._num_iters = 0
        self._floor_count = 0
        self._phase = LRPhase.WARMUP if warmup_iter > 0 else LRPhase.ANNEALING
        _dbg(
            "OVERRIDE_APPLIED",
            f"start_lr={start_lr} warmup={warmup_iter} total={total_iters} "
            f"decay={decay_style} min_lr={min_lr}",
        )

    def restore_from_checkpoint(self, sd: Dict[str, Any]) -> None:
        """
        对应 --use-checkpoint-lr-scheduler：
        完全以 checkpoint state_dict 为准。
        语义上等价于 load_state_dict(sd) with use_checkpoint=True。
        """
        _dbg("RESTORE_FROM_CKPT", f"iter={sd.get('num_iters','?')} phase={sd.get('phase','?')}")
        old_flag = self.use_checkpoint_lr_scheduler
        self.use_checkpoint_lr_scheduler = True
        self.load_state_dict(sd)
        self.use_checkpoint_lr_scheduler = old_flag
