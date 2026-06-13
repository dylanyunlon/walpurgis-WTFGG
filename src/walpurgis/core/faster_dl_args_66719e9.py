# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: Megatron-LM commit 66719e973
# 原标题: Faster dataloader merge (#1)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 涉及上游文件：
#   - arguments.py      → 新增 --shuffle / --presplit-sentences 两个参数
#   - configure_data.py → make_data_loader 使用新 RandomSampler；
#                         make_tfrecord_loaders 修复 num_workers/threaded_dl
#   - pretrain_bert.py  → 新增 args.shuffle 传递（一行改动）
#
# 「横眉冷对千夫指，俯首甘为孺子牛。」
# —— 鲁迅《自嘲》
#
# 上游 arguments.py 的故事：一个参数的增删，往往意味着一次认知升级。
# --shuffle 的出现宣告：\"随机\"不再是运行时副产品，而是有种子、有 epoch、
# 可复现的「确定性随机」——横眉冷对不可复现的训练，俯首细耕可审计的随机策略。
# --presplit-sentences 则更彻底：把运行时的代价前移到预处理阶段，
# 是「把将来的苦难放在今天受」的工程哲学——今日多花一次 CPU，
# 换来每个 epoch 的零代价句子迭代。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. `ArgumentGroupSpec` dataclass — 将上游两个新 add_argument 调用
#      封装为可序列化的参数规格，`to_argparse_kwargs()` 返回 add_argument 所需 dict，
#      `compatibility_note()` 记录每个参数的版本来源（66719e973），
#      使参数演化历史在代码中可追溯，而非只存在于 git log 里。
#   2. `DataLoaderPolicy` 枚举 + `DataLoaderPolicyResolver` — 将
#      configure_data.py 中的采样器选择逻辑（shuffle → RandomSampler,
#      not shuffle → SequentialSampler）建模为策略枚举，
#      `resolve(shuffle, dataset_size, batch_size, train_iters)` 返回完整策略描述，
#      上游仅有裸 `if shuffle: sampler = ...` 无任何策略文档化。
#   3. 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖所有决策路径。

import os as _os
import sys as _sys
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DL-ARGS:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# ArgumentGroupSpec — 上游新增参数的可序列化规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgumentGroupSpec:
    """单个 argparse 参数的完整规格描述。

    上游 arguments.py 中 `add_argument_group('data', ...)` 新增了两个参数：
    - ``--shuffle``：确定性随机打乱（seed + epoch 绑定）
    - ``--presplit-sentences``：预分割句子格式

    Walpurgis 将每个参数的元数据（flag / action / help / 来源 commit）
    封装为不可变 dataclass，使参数演化历史在代码中可追溯，
    而非依赖 ``git log arguments.py`` 才能回溯。

    Args:
        flag: 参数名（如 ``--shuffle``）。
        action: argparse action（如 ``store_true``）。
        help_text: 参数说明文字（上游原文）。
        source_commit: 引入此参数的 Megatron-LM commit hash。
        walpurgis_note: Walpurgis 对此参数的工程意义注释。
    """
    flag: str
    action: str
    help_text: str
    source_commit: str
    walpurgis_note: str = ""

    def to_argparse_kwargs(self) -> Dict[str, Any]:
        """生成可直接传入 ``add_argument()`` 的 kwargs 字典。"""
        kwargs: Dict[str, Any] = {"action": self.action, "help": self.help_text}
        _dbg("ARG_SPEC", f"flag={self.flag!r} kwargs={kwargs}")
        return kwargs

    def compatibility_note(self) -> str:
        """生成兼容性说明，含 commit 来源与 Walpurgis 注释。"""
        note = (
            f"  参数: {self.flag}\n"
            f"  来源: Megatron-LM {self.source_commit}\n"
            f"  action: {self.action}\n"
            f"  help: {self.help_text}\n"
            f"  Walpurgis: {self.walpurgis_note or '（无额外注释）'}"
        )
        return note


# ---------------------------------------------------------------------------
# 上游两个新增参数的规格实例
# ---------------------------------------------------------------------------

SHUFFLE_ARG_SPEC = ArgumentGroupSpec(
    flag="--shuffle",
    action="store_true",
    help_text=(
        "Shuffle data. Shuffling is deterministic "
        "based on seed and current epoch."
    ),
    source_commit="66719e973",
    walpurgis_note=(
        "配合 data_utils/samplers.py::RandomSampler 使用，"
        "种子计算为 seed × epoch，训练可完全复现。"
        "对应 configure_data.py 的采样器选择策略：shuffle=True → RandomSampler, "
        "shuffle=False → SequentialSampler。"
    ),
)

PRESPLIT_SENTENCES_ARG_SPEC = ArgumentGroupSpec(
    flag="--presplit-sentences",
    action="store_true",
    help_text=(
        "Dataset content consists of documents where "
        "each document consists of newline separated sentences"
    ),
    source_commit="66719e973",
    walpurgis_note=(
        "预分割版本需配合 scripts/presplit_sentences_json.py 预处理。"
        "presplit=True 时文档以 \\n 分割存储，训练时逐句拼接，"
        "无需每 epoch 实时分词，显著提升数据加载吞吐。"
        "对应 data_utils/datasets.py 的两个分支路径。"
    ),
)

# 参数组全集（便于批量注册或审计）
FASTER_DL_ARG_SPECS = (SHUFFLE_ARG_SPEC, PRESPLIT_SENTENCES_ARG_SPEC)


# ---------------------------------------------------------------------------
# DataLoaderPolicy 枚举 — 采样器策略建模
# ---------------------------------------------------------------------------

class DataLoaderPolicy(Enum):
    """DataLoader 采样器策略枚举。

    上游 configure_data.py make_data_loader 函数的采样器选择逻辑：
        shuffle=True  → data_utils.samplers.RandomSampler(replacement=True, num_samples=...)
        shuffle=False → torch.utils.data.SequentialSampler(dataset)

    Walpurgis 将此二态显式建模：
    - RANDOM_DETERMINISTIC：有放回确定性随机采样（配合 seed+epoch）；
    - SEQUENTIAL：顺序遍历数据集一次。

    策略值格式：(sampler_class_name, is_replacement, requires_num_samples)
    """
    RANDOM_DETERMINISTIC = ("RandomSampler", True, True)
    SEQUENTIAL = ("SequentialSampler", False, False)

    @property
    def sampler_class_name(self) -> str:
        return self.value[0]

    @property
    def is_replacement(self) -> bool:
        return self.value[1]

    @property
    def requires_num_samples(self) -> bool:
        return self.value[2]

    def describe(self) -> str:
        if self == DataLoaderPolicy.RANDOM_DETERMINISTIC:
            return (
                "RANDOM_DETERMINISTIC: 有放回确定性随机采样。"
                "num_samples=batch_size×train_iters，seed×epoch 重置 Generator。"
                "多 epoch 训练下完全可复现（上游 66719e973 核心改动）。"
            )
        return (
            "SEQUENTIAL: 顺序遍历数据集，无随机性。"
            "适合评估或不需要打乱的场景。"
        )


# ---------------------------------------------------------------------------
# DataLoaderPolicyResolver — 采样器策略决策器
# ---------------------------------------------------------------------------

class DataLoaderPolicyResolver:
    """根据 args 决定采样器策略，并生成 sampler 构造参数。

    上游 configure_data.py make_data_loader 的核心逻辑（精简）：
        if shuffle:
            sampler = data_utils.samplers.RandomSampler(
                dataset, replacement=True,
                num_samples=batch_size*args.train_iters)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    Walpurgis 将此决策封装为 `resolve()`，返回 (policy, sampler_kwargs)，
    使 configure_data.py 的条件分支从隐式变为显式可测试。
    """

    @staticmethod
    def resolve(
        shuffle: bool,
        dataset_size: int,
        batch_size: int,
        train_iters: int,
    ) -> Tuple[DataLoaderPolicy, Dict[str, Any]]:
        """决定采样策略并返回 sampler 构造参数。

        Args:
            shuffle: 是否打乱（对应 ``--shuffle`` flag）。
            dataset_size: 数据集大小。
            batch_size: 每批大小。
            train_iters: 训练总迭代数。

        Returns:
            (policy, sampler_kwargs) 二元组：
            - policy: DataLoaderPolicy 枚举值；
            - sampler_kwargs: 传入 sampler 构造函数的 kwargs 字典。
        """
        if shuffle:
            policy = DataLoaderPolicy.RANDOM_DETERMINISTIC
            num_samples = batch_size * train_iters
            kwargs: Dict[str, Any] = {
                "replacement": True,
                "num_samples": num_samples,
            }
            _dbg(
                "POLICY_RESOLVE",
                f"shuffle=True → {policy.sampler_class_name} "
                f"num_samples={num_samples} "
                f"(batch_size={batch_size} × train_iters={train_iters})"
            )
        else:
            policy = DataLoaderPolicy.SEQUENTIAL
            kwargs = {}
            _dbg(
                "POLICY_RESOLVE",
                f"shuffle=False → {policy.sampler_class_name}"
            )

        return policy, kwargs

    @staticmethod
    def policy_summary(policy: DataLoaderPolicy, kwargs: Dict[str, Any]) -> str:
        """生成可读的策略摘要（用于日志）。"""
        return (
            f"Policy: {policy.name}\n"
            f"  sampler: {policy.sampler_class_name}\n"
            f"  kwargs:  {kwargs}\n"
            f"  描述:    {policy.describe()}"
        )


# ---------------------------------------------------------------------------
# TFRecordArgsPolicy — make_tfrecord_loaders 参数修复策略
# ---------------------------------------------------------------------------

@dataclass
class TFRecordArgsPolicy:
    """TFRecordDataLoader 构造参数策略，封装 configure_data.py 两处修复。

    上游 66719e973 对 make_tfrecord_loaders 做了两处关键修复：
        原版：``'num_workers': args.num_workers``
        修复：``'num_workers': max(args.num_workers, 1)``
            — 当用户传 0 时 TFRecord reader 无法工作，此修复保证最低 1 个 worker

        原版：``'seed': args.seed+args.rank+1``（无 threaded_dl 参数）
        修复：``'seed': args.seed + args.rank + 1,``
              ``'threaded_dl': args.num_workers > 0``
            — 新增 threaded_dl 键，控制 ThreadedIterator 是否启用

    Walpurgis 将这两处修复封装为 `build_loader_args()`，
    使决策逻辑集中、可复测，而非散落在 configure_data.py 的字面量 dict 中。

    Args:
        num_workers: 用户原始 ``--num-workers`` 值。
        seed: 基准随机种子。
        rank: 当前进程 rank。
        seq_length: 序列最大长度（``--seq-length``）。
        max_preds_per_seq: 每序列最大预测数（``--max-preds-per-seq``）。
        vocab_size: 词表大小。
    """
    num_workers: int
    seed: int
    rank: int
    seq_length: int = 512
    max_preds_per_seq: int = 80
    vocab_size: int = 30522

    def build_loader_args(self, train: bool = True) -> Dict[str, Any]:
        """生成传入 TFRecordDataLoader 的完整参数字典。

        修复点：
        - ``num_workers`` → ``max(num_workers, 1)``（上游 fix #1）
        - 新增 ``threaded_dl``（上游 fix #2）
        - ``seed`` → ``seed + rank + 1``（rank 偏移，保证各进程独立）

        Args:
            train: 是否为训练集（控制 ``train`` 字段）。

        Returns:
            可直接传入 ``TFRecordDataLoader(**args)`` 的字典。
        """
        effective_workers = max(self.num_workers, 1)
        threaded_dl = self.num_workers > 0
        rank_seed = self.seed + self.rank + 1

        args_dict: Dict[str, Any] = {
            "max_seq_len": self.seq_length,
            "max_preds_per_seq": self.max_preds_per_seq,
            "train": train,
            "num_workers": effective_workers,
            "seed": rank_seed,
            "threaded_dl": threaded_dl,
        }

        _dbg(
            "TF_ARGS_BUILD",
            f"train={train} effective_workers={effective_workers} "
            f"threaded_dl={threaded_dl} rank_seed={rank_seed} "
            f"args={args_dict}"
        )
        return args_dict

    def audit(self) -> str:
        """输出参数决策审计报告。"""
        train_args = self.build_loader_args(train=True)
        val_args = self.build_loader_args(train=False)
        lines = [
            "=== TFRecordArgsPolicy Audit ===",
            f"  input:  num_workers={self.num_workers} seed={self.seed} rank={self.rank}",
            f"  fix#1:  effective_workers = max({self.num_workers}, 1) = {max(self.num_workers, 1)}",
            f"  fix#2:  threaded_dl = {self.num_workers} > 0 = {self.num_workers > 0}",
            f"  fix#3:  rank_seed = {self.seed} + {self.rank} + 1 = {self.seed + self.rank + 1}",
            f"  train_args:  {train_args}",
            f"  val_args:    {val_args}",
            "================================",
        ]
        report = "\n".join(lines)
        _dbg("TF_ARGS_AUDIT", "\n" + report)
        return report


# ---------------------------------------------------------------------------
# ArgumentAuditReport — 全部新增参数的审计报告
# ---------------------------------------------------------------------------

@dataclass
class ArgumentAuditReport:
    """66719e973 新增参数的完整审计报告。

    Args:
        specs: 参数规格列表。
    """
    specs: Tuple[ArgumentGroupSpec, ...] = field(
        default_factory=lambda: FASTER_DL_ARG_SPECS
    )

    def report(self) -> str:
        """生成多行文本审计报告。"""
        lines = [
            "=== Megatron-LM 66719e973 Faster DataLoader 参数审计 ===",
            f"  新增参数数量: {len(self.specs)}",
            "",
        ]
        for i, spec in enumerate(self.specs, 1):
            lines.append(f"  [{i}] {spec.compatibility_note()}")
            lines.append("")

        lines += [
            "  核心改动摘要：",
            "  1. --shuffle 启用确定性随机采样（RandomSampler + seed×epoch）",
            "  2. --presplit-sentences 启用预分割句子格式，避免 epoch 内重复分词",
            "  3. configure_data.py: shuffle=True 时使用新 RandomSampler",
            "  4. configure_data.py: TFRecord loader 修复 num_workers/threaded_dl",
            "=======================================================",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """模块加载时的五项断言（WALPURGIS_DEBUG=1 时执行）。"""
    if not _DEBUG:
        return

    # 1. ArgumentGroupSpec to_argparse_kwargs
    spec = SHUFFLE_ARG_SPEC
    kwargs = spec.to_argparse_kwargs()
    assert kwargs["action"] == "store_true"
    assert "seed" in kwargs["help"] or "Shuffle" in kwargs["help"]
    _dbg("SELF_CHECK", f"ArgumentGroupSpec OK: {spec.flag}")

    # 2. FASTER_DL_ARG_SPECS 包含两个参数
    assert len(FASTER_DL_ARG_SPECS) == 2
    flags = [s.flag for s in FASTER_DL_ARG_SPECS]
    assert "--shuffle" in flags and "--presplit-sentences" in flags
    _dbg("SELF_CHECK", f"FASTER_DL_ARG_SPECS OK: {flags}")

    # 3. DataLoaderPolicyResolver.resolve — shuffle=True
    policy, kwargs = DataLoaderPolicyResolver.resolve(
        shuffle=True, dataset_size=1000, batch_size=8, train_iters=100
    )
    assert policy == DataLoaderPolicy.RANDOM_DETERMINISTIC
    assert kwargs["replacement"] is True
    assert kwargs["num_samples"] == 800
    _dbg("SELF_CHECK", f"PolicyResolver(shuffle=True) OK: num_samples={kwargs['num_samples']}")

    # 4. DataLoaderPolicyResolver.resolve — shuffle=False
    policy2, kwargs2 = DataLoaderPolicyResolver.resolve(
        shuffle=False, dataset_size=500, batch_size=4, train_iters=50
    )
    assert policy2 == DataLoaderPolicy.SEQUENTIAL
    assert kwargs2 == {}
    _dbg("SELF_CHECK", "PolicyResolver(shuffle=False) OK")

    # 5. TFRecordArgsPolicy.build_loader_args — num_workers=0 修复
    tf_policy = TFRecordArgsPolicy(num_workers=0, seed=42, rank=1)
    args = tf_policy.build_loader_args(train=True)
    assert args["num_workers"] == 1, f"expected 1, got {args['num_workers']}"
    assert args["threaded_dl"] is False
    assert args["seed"] == 44  # 42 + 1 + 1
    assert args["train"] is True
    _dbg("SELF_CHECK", f"TFRecordArgsPolicy(num_workers=0) OK: {args}")

    _dbg("SELF_CHECK", "✓ 全部 5 项断言通过")


_dbg("MODULE_LOAD", "faster_dl_args_66719e9 载入开始")
_self_check()
_dbg("MODULE_LOAD", "faster_dl_args_66719e9 载入完成")
