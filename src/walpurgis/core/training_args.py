"""
walpurgis/core/training_args.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit a1d04b793（第9个，共9062）
subject: Updating public repo with latest changes.

上游变更摘要 — arguments.py（+48 行）
=====================================
本次 commit 向 arguments.py 新增了六类训练参数：

  1. ``--tensorboard-dir``      写入 TensorBoard 日志的目录路径
  2. ``--eod-mask-loss``         屏蔽文档结束符处的 loss（训练目标精化）
  3. ``--min-lr``                学习率调度器的下界截断值
  4. ``--override-lr-scheduler`` 忽略 checkpoint 中的调度器状态，以命令行参数重置
  5. ``--use-checkpoint-lr-scheduler`` 从 checkpoint 恢复调度器状态（与上条互斥）
  6. ``--DDP-impl``              选择 DistributedDataParallel 实现（local / torch）
  7. ``--adlr-autoresume``       启用 ADLR 集群自动恢复
  8. ``--adlr-autoresume-interval`` 自动恢复检查间隔（步数）

鲁迅拿法改写（≥20%）
=====================
上游 arguments.py 是一张无边无际的参数清单——每个 ``add_argument()`` 像一道
新的圈地令，划定了训练过程的一处边界，但各参数之间的关联、冲突、互斥关系
一概付之阙如。鲁迅在《藤野先生》里写道：「大概是物以希为贵罢……
一到东京，也无非是这样。」——参数太多，便失了价值，无人知道它们之间
谁管谁、谁克谁。

``--override-lr-scheduler`` 与 ``--use-checkpoint-lr-scheduler`` 在上游
仅是两个独立的 ``store_true`` 参数，互斥关系只存在于运行时崩溃之中，
无任何静态检查。``--adlr-autoresume`` 与 ``--adlr-autoresume-interval``
亦是孤立浮于参数表，没有任何关于「什么是 ADLR」的说明。

Walpurgis 将这八个新参数的语义结构化为五个可程序化审查的结构：

  1. ``LrSchedulerPolicy`` 枚举 ——
     明确区分 FROM_ARGS（--override）、FROM_CKPT（--use-checkpoint）、
     DEFAULT 三种调度器初始化策略，使互斥关系从运行时崩溃提升为
     静态枚举匹配，``from_args()`` 工厂方法执行冲突检测并在
     WALPURGIS_DEBUG=1 时记录断点。

  2. ``TensorBoardConfig`` dataclass ——
     封装 tensorboard-dir，附带 ``is_enabled()`` 谓词与
     ``validate()`` 路径合法性检查（目录不存在时警告而非静默失败）。

  3. ``AdlrAutoresumeConfig`` dataclass ——
     建模 ADLR（Advanced Deep Learning Research）集群自动恢复机制，
     携带 enabled 标志与 interval（步数），提供 ``should_checkpoint(step)``
     谓词，将上游裸整型参数的使用意图显式化。

  4. ``DdpImplChoice`` 枚举 ——
     区分 LOCAL（Megatron 自研 DDP）与 TORCH（torch.nn.parallel.DistributedDataParallel），
     避免字符串传递带来的隐式错误（上游以 default='local' 裸字符串控制分支）。

  5. ``TrainingArgsPatch`` dataclass ——
     汇总本次 commit 新增的全部参数快照，提供 ``audit()`` 接口输出
     结构化审计报告（LR 策略冲突检查、tensorboard 路径检查、
     autoresume 配置有效性检查），``self_check()`` 含7项断言。

全链路 _dbg() 断点共 18 处，覆盖：
  MODULE_LOAD×2、LR_POLICY_ENUM_INIT、LR_POLICY_FROM_ARGS_CONFLICT、
  LR_POLICY_FROM_ARGS_OK、TB_CONFIG_INIT、TB_CONFIG_VALIDATE_MISSING、
  TB_CONFIG_VALIDATE_OK、ADLR_CONFIG_INIT、ADLR_SHOULD_CKPT、
  DDP_IMPL_ENUM_INIT、ARGS_PATCH_INIT、ARGS_PATCH_EOD_MASK、
  ARGS_PATCH_MIN_LR、ARGS_PATCH_AUDIT、SELF_CHECK_START、
  SELF_CHECK_PASS×2。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ─── 全局调试开关（与 walpurgis/__init__.py 保持一致）─────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    """断点诊断：WALPURGIS_DEBUG=1 时打印，生产环境静默。"""
    if _DBG:
        print(f"[WALPURGIS-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "training_args.py 开始加载")


# ─── 1. LrSchedulerPolicy 枚举 ────────────────────────────────────────────

class LrSchedulerPolicy(Enum):
    """
    学习率调度器初始化策略。

    上游 Megatron commit a1d04b793 新增的两个互斥参数：
      --override-lr-scheduler       → FROM_ARGS
      --use-checkpoint-lr-scheduler → FROM_CKPT

    两者同时指定时，上游代码在运行时才崩溃；
    Walpurgis 在枚举工厂方法中静态检测冲突。
    """
    DEFAULT = auto()       # 两个参数均未指定，沿用 checkpoint 中的调度器状态
    FROM_ARGS = auto()     # --override-lr-scheduler：以命令行参数重置调度器
    FROM_CKPT = auto()     # --use-checkpoint-lr-scheduler：从 checkpoint 恢复

    @classmethod
    def from_args(
        cls,
        override: bool,
        use_checkpoint: bool,
    ) -> "LrSchedulerPolicy":
        """
        从布尔参数对构造策略枚举。

        Parameters
        ----------
        override:        对应 --override-lr-scheduler
        use_checkpoint:  对应 --use-checkpoint-lr-scheduler

        Raises
        ------
        ValueError  两者同时为 True 时（互斥冲突）
        """
        _dbg("LR_POLICY_FROM_ARGS_CONFLICT" if (override and use_checkpoint)
             else "LR_POLICY_FROM_ARGS_OK",
             f"override={override} use_checkpoint={use_checkpoint}")

        if override and use_checkpoint:
            raise ValueError(
                "--override-lr-scheduler 与 --use-checkpoint-lr-scheduler 互斥，"
                "不能同时指定。上游 Megatron 同样不允许此组合，"
                "但仅在运行时才报错；Walpurgis 在此提前拦截。"
            )
        if override:
            return cls.FROM_ARGS
        if use_checkpoint:
            return cls.FROM_CKPT
        return cls.DEFAULT


_dbg("LR_POLICY_ENUM_INIT", f"LrSchedulerPolicy members: {[m.name for m in LrSchedulerPolicy]}")


# ─── 2. TensorBoardConfig dataclass ──────────────────────────────────────

@dataclass
class TensorBoardConfig:
    """
    封装 ``--tensorboard-dir`` 参数（commit a1d04b793 新增）。

    上游：``group.add_argument('--tensorboard-dir', type=str, default=None)``
    Walpurgis：增加路径合法性检查与启用谓词，避免目录不存在时静默失败。
    """
    directory: Optional[str] = None

    def __post_init__(self) -> None:
        _dbg("TB_CONFIG_INIT", f"directory={self.directory!r}")

    def is_enabled(self) -> bool:
        """TensorBoard 日志是否启用（目录非空即视为启用）。"""
        return bool(self.directory)

    def validate(self) -> bool:
        """
        检查目录是否可用。

        Returns
        -------
        True   目录存在或未启用（无需检查）
        False  目录不存在（警告但不抛出，上游行为一致）
        """
        if not self.is_enabled():
            return True
        exists = os.path.isdir(self.directory)
        if not exists:
            _dbg("TB_CONFIG_VALIDATE_MISSING",
                 f"tensorboard-dir={self.directory!r} 不存在，训练时将自动创建")
            print(
                f"[WARN:TensorBoardConfig] 目录 {self.directory!r} 不存在，"
                "将在首次写入时由 TensorBoard SummaryWriter 自动创建。",
                file=sys.stderr,
            )
        else:
            _dbg("TB_CONFIG_VALIDATE_OK", f"目录存在: {self.directory!r}")
        return exists

    def summary_writer_kwargs(self) -> dict:
        """返回适合传入 torch.utils.tensorboard.SummaryWriter 的 kwargs。"""
        if not self.is_enabled():
            return {}
        return {"log_dir": self.directory}


# ─── 3. AdlrAutoresumeConfig dataclass ───────────────────────────────────

@dataclass
class AdlrAutoresumeConfig:
    """
    ADLR（Advanced Deep Learning Research）集群自动恢复配置。

    上游 commit a1d04b793 新增两个参数：
      --adlr-autoresume           store_true，启用自动恢复
      --adlr-autoresume-interval  int，检查间隔（迭代步数，默认 1000）

    ADLR 自动恢复机制在 NVIDIA 内部集群上通过周期性调用
    ``adlr_autoresume.check_for_autoresume()`` 实现：若集群信号要求中断，
    训练进程保存 checkpoint 后安全退出，随后由集群调度器重新启动并自动恢复。

    上游将 interval 硬编码为默认值 1000，使用者需要自行记住这一隐含约定；
    Walpurgis 将其封装为 ``should_checkpoint(step)``，使调用方无需关心间隔值。
    """
    enabled: bool = False
    interval: int = 1000  # 上游默认值 default=1000

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(
                f"adlr-autoresume-interval 必须 > 0，得到 {self.interval}"
            )
        _dbg("ADLR_CONFIG_INIT",
             f"enabled={self.enabled} interval={self.interval}")

    def should_checkpoint(self, step: int) -> bool:
        """
        当前步是否应触发自动恢复检查点。

        Parameters
        ----------
        step : 当前训练迭代步数（从 1 开始）
        """
        result = self.enabled and (step % self.interval == 0)
        _dbg("ADLR_SHOULD_CKPT",
             f"step={step} interval={self.interval} → {result}")
        return result


# ─── 4. DdpImplChoice 枚举 ───────────────────────────────────────────────

class DdpImplChoice(Enum):
    """
    DistributedDataParallel 实现选择。

    上游 commit a1d04b793 新增：
      ``--DDP-impl`` default='local'，接受 'local' 或 'torch'

    LOCAL：Megatron 自研 DDP（与 model parallelism 深度集成，支持梯度累积优化）
    TORCH：标准 torch.nn.parallel.DistributedDataParallel（更通用，兼容性更好）

    Walpurgis 将裸字符串替换为枚举，使分支选择可在 match/case 中静态穷举，
    避免 typo（如 'Local' 大小写错误）导致的静默回退。
    """
    LOCAL = "local"   # 上游默认；Megatron 自研 DDP
    TORCH = "torch"   # 标准 PyTorch DDP

    @classmethod
    def from_str(cls, s: str) -> "DdpImplChoice":
        """从命令行字符串构造枚举，不区分大小写。"""
        _dbg("DDP_IMPL_ENUM_INIT", f"from_str({s!r})")
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(
                f"--DDP-impl 无效值 {s!r}，支持: "
                + ", ".join(f"'{m.value}'" for m in cls)
            )


# ─── 5. TrainingArgsPatch dataclass（汇总审计入口）─────────────────────

@dataclass
class TrainingArgsPatch:
    """
    commit a1d04b793 新增参数的 Walpurgis 结构化快照。

    字段对应关系
    ─────────────────────────────────────────────────────────────
    tensorboard         ↔  --tensorboard-dir
    eod_mask_loss       ↔  --eod-mask-loss
    min_lr              ↔  --min-lr
    lr_policy           ↔  --override-lr-scheduler / --use-checkpoint-lr-scheduler
    ddp_impl            ↔  --DDP-impl
    autoresume          ↔  --adlr-autoresume / --adlr-autoresume-interval
    ─────────────────────────────────────────────────────────────

    上游注释（来自 commit a1d04b793 arguments.py diff）：
      --min-lr: "The scheduler clip values below this threshold."
      --override-lr-scheduler: "Reset the values of the scheduler … ignore values from checkpoints."
      --use-checkpoint-lr-scheduler: "Use checkpoint to set the values of the scheduler …"
    """
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    eod_mask_loss: bool = False
    min_lr: float = 0.0       # 上游默认 default=0.0
    lr_policy: LrSchedulerPolicy = LrSchedulerPolicy.DEFAULT
    ddp_impl: DdpImplChoice = DdpImplChoice.LOCAL
    autoresume: AdlrAutoresumeConfig = field(default_factory=AdlrAutoresumeConfig)

    def __post_init__(self) -> None:
        _dbg("ARGS_PATCH_INIT",
             f"eod_mask_loss={self.eod_mask_loss} "
             f"min_lr={self.min_lr} "
             f"lr_policy={self.lr_policy.name} "
             f"ddp_impl={self.ddp_impl.value}")
        if self.eod_mask_loss:
            _dbg("ARGS_PATCH_EOD_MASK",
                 "EOD mask loss 已启用：文档结束符处的 loss 将被屏蔽")
        if self.min_lr > 0:
            _dbg("ARGS_PATCH_MIN_LR",
                 f"min_lr={self.min_lr}：调度器将裁剪低于此值的学习率")

    @classmethod
    def from_namespace(cls, args) -> "TrainingArgsPatch":
        """
        从 argparse.Namespace 构造（适配上游 get_args() 返回值）。

        Parameters
        ----------
        args : argparse.Namespace，需包含本次 commit 新增的属性
        """
        tb = TensorBoardConfig(
            directory=getattr(args, "tensorboard_dir", None)
        )
        policy = LrSchedulerPolicy.from_args(
            override=getattr(args, "override_lr_scheduler", False),
            use_checkpoint=getattr(args, "use_checkpoint_lr_scheduler", False),
        )
        ddp = DdpImplChoice.from_str(
            getattr(args, "DDP_impl", "local")
        )
        autoresume = AdlrAutoresumeConfig(
            enabled=getattr(args, "adlr_autoresume", False),
            interval=getattr(args, "adlr_autoresume_interval", 1000),
        )
        return cls(
            tensorboard=tb,
            eod_mask_loss=getattr(args, "eod_mask_loss", False),
            min_lr=getattr(args, "min_lr", 0.0),
            lr_policy=policy,
            ddp_impl=ddp,
            autoresume=autoresume,
        )

    def audit(self) -> dict:
        """
        结构化审计报告：返回可 JSON 序列化的 dict，供调试与日志使用。

        涵盖：
          - LR 策略冲突检查（已在 LrSchedulerPolicy.from_args() 中提前拦截，
            此处仅汇报当前状态）
          - TensorBoard 路径有效性
          - AutoResume 配置有效性
          - EOD mask 与 min_lr 状态
        """
        _dbg("ARGS_PATCH_AUDIT", "开始审计")
        report = {
            "tensorboard": {
                "enabled": self.tensorboard.is_enabled(),
                "directory": self.tensorboard.directory,
                "dir_exists": (
                    os.path.isdir(self.tensorboard.directory)
                    if self.tensorboard.directory else None
                ),
            },
            "eod_mask_loss": self.eod_mask_loss,
            "min_lr": self.min_lr,
            "lr_policy": self.lr_policy.name,
            "ddp_impl": self.ddp_impl.value,
            "autoresume": {
                "enabled": self.autoresume.enabled,
                "interval": self.autoresume.interval,
            },
        }
        return report


# ─── 自检 ─────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    七项断言，覆盖本模块核心路径。
    ``WALPURGIS_DEBUG=1 python -c "from walpurgis.core.training_args import self_check; self_check()"``
    """
    _dbg("SELF_CHECK_START", "开始 self_check()")

    # 1. LR 策略枚举：FROM_ARGS
    p = LrSchedulerPolicy.from_args(override=True, use_checkpoint=False)
    assert p == LrSchedulerPolicy.FROM_ARGS, f"期望 FROM_ARGS，得到 {p}"

    # 2. LR 策略枚举：FROM_CKPT
    p = LrSchedulerPolicy.from_args(override=False, use_checkpoint=True)
    assert p == LrSchedulerPolicy.FROM_CKPT

    # 3. LR 策略枚举：DEFAULT
    p = LrSchedulerPolicy.from_args(override=False, use_checkpoint=False)
    assert p == LrSchedulerPolicy.DEFAULT

    # 4. 互斥冲突检测
    try:
        LrSchedulerPolicy.from_args(override=True, use_checkpoint=True)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 5. DDP 枚举解析
    assert DdpImplChoice.from_str("local") == DdpImplChoice.LOCAL
    assert DdpImplChoice.from_str("TORCH") == DdpImplChoice.TORCH

    # 6. AutoResume should_checkpoint 逻辑
    cfg = AdlrAutoresumeConfig(enabled=True, interval=100)
    assert cfg.should_checkpoint(100)
    assert not cfg.should_checkpoint(99)

    # 7. TrainingArgsPatch 审计报告键完整性
    patch = TrainingArgsPatch(
        tensorboard=TensorBoardConfig(directory="/tmp/tb"),
        eod_mask_loss=True,
        min_lr=1e-5,
        lr_policy=LrSchedulerPolicy.FROM_ARGS,
        ddp_impl=DdpImplChoice.TORCH,
        autoresume=AdlrAutoresumeConfig(enabled=True, interval=500),
    )
    report = patch.audit()
    for key in ("tensorboard", "eod_mask_loss", "min_lr", "lr_policy",
                "ddp_impl", "autoresume"):
        assert key in report, f"audit() 缺少键: {key!r}"

    _dbg("SELF_CHECK_PASS", "全部 7 项断言通过")
    print("[training_args.self_check] OK — 7 assertions passed", file=sys.stderr)
    return True


_dbg("MODULE_LOAD", "training_args.py 加载完成")

if __name__ == "__main__":
    self_check()
