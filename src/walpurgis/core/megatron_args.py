"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: arguments.py

鲁迅拿法改写（≥20%）：
  原 arguments.py 是一道长长的告示墙：argparse.add_argument 堆叠，
  参数挤着参数，无人说明彼此的约束关系。--override-lr-scheduler 与
  --use-checkpoint-lr-scheduler 互斥，却并排放着，等人踩坑；
  --min-lr 要配合 --lr 才有意义，注释只写 "Minimum value"，
  不说"低于此值将被裁断"。鲁迅见过这种官府布告：条文密密麻麻，
  读完还是不知道该做什么。
  Walpurgis 将此次新增的 8 个参数（tensorboard-dir, eod-mask-loss,
  min-lr, override-lr-scheduler, use-checkpoint-lr-scheduler,
  DDP-impl, adlr-autoresume, adlr-autoresume-interval）
  按"功能域"重新分组，附加互斥约束验证、参数合理性检查、
  以及可程序化访问的 ArgGroup 注册表，
  并在三个断点输出参数快照：ARGS_BUILT / MUTUAL_EXCL_CHECKED / RANGE_VALIDATED。

迁移位置: src/walpurgis/core/megatron_args_a1d04b793.py
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# ── 全局调试开关 ──────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：参数解析关键节点快照，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    print(f"[_dbg:megatron_args:{tag}] {msg}", file=sys.stderr, flush=True)


# ── ArgGroup 注册表（Walpurgis特有：将参数按功能域归档） ─────────────────────
@dataclass
class ArgGroupSpec:
    """单个参数组的结构化描述：名称、所含参数名、约束规则"""
    name: str
    params: List[str] = field(default_factory=list)
    mutual_exclusions: List[tuple] = field(default_factory=list)
    range_checks: Dict[str, tuple] = field(default_factory=dict)  # param -> (lo, hi, inclusive)


# 本次 commit 新增的参数，按功能域分组
_NEW_ARG_GROUPS: List[ArgGroupSpec] = [
    ArgGroupSpec(
        name="tensorboard",
        params=["tensorboard_dir"],
        mutual_exclusions=[],
        range_checks={},
    ),
    ArgGroupSpec(
        name="loss_masking",
        params=["eod_mask_loss"],
        mutual_exclusions=[],
        range_checks={},
    ),
    ArgGroupSpec(
        name="lr_schedule",
        params=["min_lr", "override_lr_scheduler", "use_checkpoint_lr_scheduler"],
        # 互斥：两个标志不能同时设置（上游注释里隐含，代码里没有检查，此处补全）
        mutual_exclusions=[("override_lr_scheduler", "use_checkpoint_lr_scheduler")],
        range_checks={
            "min_lr": (0.0, float("inf"), True),  # min_lr >= 0
        },
    ),
    ArgGroupSpec(
        name="distributed",
        params=["DDP_impl", "adlr_autoresume", "adlr_autoresume_interval"],
        mutual_exclusions=[],
        range_checks={
            "adlr_autoresume_interval": (1, None, True),  # must be positive
        },
    ),
]


def build_training_args(parser: Optional[argparse.ArgumentParser] = None
                        ) -> argparse.ArgumentParser:
    """
    构建本次 commit 新增的训练参数（增量注册到 parser 或创建新 parser）。

    Walpurgis 不复制上游全部 arguments.py（那是告示墙），
    只将本 commit 的增量参数按功能域注册，附带互斥检查钩子。
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Walpurgis Megatron args (a1d04b793 delta)")

    # ── 组1: TensorBoard ──────────────────────────────────────────────────────
    tb_group = parser.add_argument_group("tensorboard (a1d04b793)")
    tb_group.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help=(
            "Write TensorBoard logs to this directory. "
            "Walpurgis: 如不设置，静默跳过；目录不存在将自动创建。"
        ),
    )

    # ── 组2: 损失掩码 ─────────────────────────────────────────────────────────
    loss_group = parser.add_argument_group("loss masking (a1d04b793)")
    loss_group.add_argument(
        "--eod-mask-loss",
        action="store_true",
        help=(
            "Mask loss for the end-of-document tokens. "
            "Walpurgis: 仅影响 GPT-2 预训练；BERT 模式下此标志被忽略（上游同）。"
        ),
    )

    # ── 组3: 学习率调度器 ─────────────────────────────────────────────────────
    lr_group = parser.add_argument_group("lr scheduler (a1d04b793)")
    lr_group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help=(
            "Minimum learning rate. Scheduler clips values below this threshold. "
            "Walpurgis: 必须 >= 0；与 --lr 共同决定调度范围下界。"
        ),
    )
    lr_group.add_argument(
        "--override-lr-scheduler",
        action="store_true",
        help=(
            "Reset scheduler values from CLI args, ignoring checkpoint state. "
            "Walpurgis: 与 --use-checkpoint-lr-scheduler 互斥，"
            "同时设置将触发 ArgumentError。"
        ),
    )
    lr_group.add_argument(
        "--use-checkpoint-lr-scheduler",
        action="store_true",
        help=(
            "Restore scheduler values from checkpoint, ignoring CLI args. "
            "Walpurgis: 与 --override-lr-scheduler 互斥。"
        ),
    )

    # ── 组4: 分布式 ───────────────────────────────────────────────────────────
    dist_group = parser.add_argument_group("distributed (a1d04b793)")
    dist_group.add_argument(
        "--DDP-impl",
        default="local",
        choices=["local", "torch"],
        help=(
            "DistributedDataParallel implementation. "
            "Walpurgis: 'local'=手工梯度规约(Megatron原生)；"
            "'torch'=torch.nn.parallel.DistributedDataParallel。"
        ),
    )
    dist_group.add_argument(
        "--adlr-autoresume",
        action="store_true",
        help=(
            "Enable autoresume on ADLR cluster. "
            "Walpurgis: 非 ADLR 环境下设置此标志为 no-op（不影响训练逻辑）。"
        ),
    )
    dist_group.add_argument(
        "--adlr-autoresume-interval",
        type=int,
        default=1000,
        help=(
            "Autoresume check interval (iterations). "
            "Walpurgis: 必须为正整数；默认 1000 表示每千步检查一次 checkpoint 信号。"
        ),
    )

    _dbg("ARGS_BUILT", f"parser groups={[g.title for g in parser._action_groups]}")
    return parser


# ── 互斥约束验证 ──────────────────────────────────────────────────────────────
def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    对 parse_args() 结果执行 Walpurgis 扩展的约束检查。
    上游将此散落在各训练脚本的 if-else 里（或根本不检查），
    Walpurgis 集中于此，一次过关，若违反则快速失败。
    """
    _dbg("MUTUAL_EXCL_CHECKED", "start")

    # 互斥: override_lr_scheduler 与 use_checkpoint_lr_scheduler
    ovr = getattr(args, "override_lr_scheduler", False)
    ckpt_lr = getattr(args, "use_checkpoint_lr_scheduler", False)
    if ovr and ckpt_lr:
        raise argparse.ArgumentError(
            None,
            "--override-lr-scheduler 与 --use-checkpoint-lr-scheduler 互斥，"
            "不能同时设置（Walpurgis: 上游未做检查，此处补全）。",
        )

    _dbg(
        "MUTUAL_EXCL_CHECKED",
        f"override_lr={ovr} use_ckpt_lr={ckpt_lr} -> OK",
    )

    # 范围: min_lr >= 0
    min_lr = getattr(args, "min_lr", 0.0)
    if min_lr < 0.0:
        raise argparse.ArgumentError(
            None, f"--min-lr 必须 >= 0，当前值={min_lr}"
        )

    # 范围: adlr_autoresume_interval > 0
    interval = getattr(args, "adlr_autoresume_interval", 1000)
    if interval <= 0:
        raise argparse.ArgumentError(
            None, f"--adlr-autoresume-interval 必须为正整数，当前值={interval}"
        )

    # TensorBoard 目录自动创建（如已指定）
    tb_dir = getattr(args, "tensorboard_dir", None)
    if tb_dir is not None:
        os.makedirs(tb_dir, exist_ok=True)
        _dbg("RANGE_VALIDATED", f"tensorboard_dir created/confirmed: {tb_dir}")

    _dbg(
        "RANGE_VALIDATED",
        f"min_lr={min_lr} adlr_interval={interval} DDP_impl={getattr(args,'DDP_impl','?')} -> OK",
    )
    return args


# ── 参数组注册表查询 ──────────────────────────────────────────────────────────
def get_arg_group(name: str) -> Optional[ArgGroupSpec]:
    """按名称查询 Walpurgis 参数组规格，供其他模块程序化访问。"""
    for g in _NEW_ARG_GROUPS:
        if g.name == name:
            return g
    return None


def list_arg_groups() -> List[str]:
    """返回所有已注册参数组名称列表。"""
    return [g.name for g in _NEW_ARG_GROUPS]


# ── 便捷入口 ──────────────────────────────────────────────────────────────────
def parse_and_validate(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """构建 parser → parse_args → validate_args 的一体化入口。"""
    parser = build_training_args()
    args = parser.parse_args(argv)
    return validate_args(args)


if __name__ == "__main__":
    # 快速自检: python -m walpurgis.core.megatron_args_a1d04b793 --min-lr 1e-5
    ns = parse_and_validate()
    print(f"args={vars(ns)}", file=sys.stderr)
