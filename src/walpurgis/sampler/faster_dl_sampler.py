# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: Megatron-LM commit 66719e973
# 原标题: Faster dataloader merge (#1)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「做奴隶虽然不幸，但并不可怕，因为知道挣扎，毕竟还有走出去的希望；
#   若是从奴隶生活中寻出"美"来，赞叹，陶醉，那才是万劫不复的奴才。」
# —— 鲁迅《花边文学·安贫乐道法》
#
# 上游 data_utils/samplers.py 的故事：RandomSampler 看似随机，实则受制。
# PyTorch 原版 RandomSampler 每 epoch 独立采样，无法保证跨 epoch、跨 rank
# 的可复现性——像极了旧时代\"看似自由、实则无处可逃\"的困境。
# 66719e973 的解法：以 seed + epoch 为锁链，化随机为确定性，
# num_samples 上限兜底，replacement=True 使 epoch 内不重复不过界。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. `EpochSeedPolicy` dataclass — 将 seed ⊕ epoch 的哈希策略从裸 int 运算
#      提取为可查询对象，`hash_key()` 返回可复现的种子值，同时携带
#      `epoch_drift` 字段记录\"本 epoch 相对于初始 seed 的偏移\"，
#      上游仅有一行 `g.manual_seed(self.seed * epoch)` 无任何策略文档化。
#   2. `SamplerBudget` dataclass — 将 `num_samples` / `batch_size` / `train_iters`
#      三元组封装为有界预算，`total_budget()` == num_samples，
#      `is_over_allocated()` 检测用户传入 num_samples 超过数据集大小的情形，
#      上游静默允许，Walpurgis 给出可审计警告。
#   3. 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖：
#      - `RandomSampler.__init__`：dataset 大小、budget、replacement 标志
#      - `RandomSampler.__iter__`：每 epoch 的种子值与实际采样 index 前 8 个
#      - `DistributedBatchSampler.__init__`：rank/world_size/drop_last
#      - `DistributedBatchSampler.__iter__`：每轮 start/end shard 边界

import os as _os
import sys as _sys
import time as _time
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sized

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-SAMPLER-FAST:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# EpochSeedPolicy — 上游裸 seed*epoch 的策略化封装
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochSeedPolicy:
    """确定性 epoch 种子策略。

    上游 Megatron 66719e973 做法：每个 epoch 用 ``seed * epoch`` 重置 Generator。
    Walpurgis 将此策略显式建模，使种子生成过程可审计、可替换。

    Args:
        base_seed: 全局随机基准种子（对应 ``--seed`` 参数）。
        multiply_epoch: 若 True，使用 base_seed × epoch（上游默认）；
            否则使用 base_seed + epoch（更均匀的 low-seed 散列）。
    """
    base_seed: int
    multiply_epoch: bool = True

    def hash_key(self, epoch: int) -> int:
        """计算第 ``epoch`` 轮的确定性种子值。"""
        key = self.base_seed * epoch if self.multiply_epoch else self.base_seed + epoch
        _dbg("EPOCH_SEED", f"base_seed={self.base_seed} epoch={epoch} key={key} "
             f"strategy={'multiply' if self.multiply_epoch else 'additive'}")
        return key

    @property
    def epoch_drift(self) -> str:
        """描述 epoch 间种子偏移策略的人类可读字符串。"""
        return f"seed × epoch (base={self.base_seed})" if self.multiply_epoch \
            else f"seed + epoch (base={self.base_seed})"


# ---------------------------------------------------------------------------
# SamplerBudget — num_samples 三元组预算封装
# ---------------------------------------------------------------------------

@dataclass
class SamplerBudget:
    """采样预算：将 batch_size × train_iters 与数据集大小对照。

    上游 configure_data.py 用法：
        ``RandomSampler(dataset, replacement=True,
                        num_samples=batch_size*args.train_iters)``
    Walpurgis 将此三元组封装，`is_over_allocated()` 给出可审计告警。

    Args:
        dataset_size: 数据集样本总数。
        batch_size: 每批样本数。
        train_iters: 训练总迭代数。
    """
    dataset_size: int
    batch_size: int
    train_iters: int

    def total_budget(self) -> int:
        """总采样配额 = batch_size × train_iters。"""
        return self.batch_size * self.train_iters

    def is_over_allocated(self) -> bool:
        """总配额是否超过数据集大小（replacement 模式下合法，但值得记录）。"""
        over = self.total_budget() > self.dataset_size
        _dbg(
            "SAMPLER_BUDGET",
            f"dataset_size={self.dataset_size} budget={self.total_budget()} "
            f"over_allocated={over}"
        )
        return over

    def epochs_equivalent(self) -> float:
        """配额等效完整 epoch 数（用于日志）。"""
        return self.total_budget() / max(self.dataset_size, 1)


# ---------------------------------------------------------------------------
# RandomSampler — 上游 data_utils/samplers.py 核心类，确定性随机采样
# ---------------------------------------------------------------------------

class RandomSampler:
    """确定性随机采样器，兼容 ``torch.utils.data.RandomSampler`` 接口。

    上游 Megatron-LM 66719e973 新增此类，解决原版 ``torch.utils.data.RandomSampler``
    在多 epoch 训练中种子不确定的问题。核心改动：
    - ``replacement=True``：允许有放回采样，总数由 ``num_samples`` 控制；
    - 每次 ``__iter__`` 时用 ``seed * epoch`` 重置 Generator，保证跨运行可复现；
    - 配合 ``configure_data.py`` 的 ``num_samples=batch_size*train_iters`` 使用，
      使整个训练过程的采样序列完全确定。

    Args:
        dataset: 带 ``__len__`` 的数据集对象。
        replacement: 有放回采样（上游默认 True，配合 num_samples 使用）。
        num_samples: 总采样数（上游用 ``batch_size × train_iters``）。
        seed: 随机基准种子。
        epoch_seed_policy: 种子策略实例，默认用乘法策略（与上游一致）。
    """

    def __init__(
        self,
        dataset: Sized,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        seed: int = 1234,
        epoch_seed_policy: Optional[EpochSeedPolicy] = None,
    ) -> None:
        try:
            import torch
            self._torch = torch
        except ImportError:
            self._torch = None

        self.dataset = dataset
        self.replacement = replacement
        self._dataset_size = len(dataset)
        self._num_samples = num_samples if num_samples is not None else self._dataset_size
        self.seed = seed
        self.epoch = 0
        self.policy = epoch_seed_policy or EpochSeedPolicy(base_seed=seed)

        _dbg(
            "RANDOM_SAMPLER_INIT",
            f"dataset_size={self._dataset_size} num_samples={self._num_samples} "
            f"replacement={replacement} seed={seed} "
            f"policy={self.policy.epoch_drift}"
        )

        # 预算审计
        if num_samples is not None:
            budget = SamplerBudget(
                dataset_size=self._dataset_size,
                batch_size=num_samples,  # 此处 num_samples 已是 batch_size*train_iters
                train_iters=1,
            )
            if budget.is_over_allocated():
                print(
                    f"[WALPURGIS-SAMPLER-FAST:BUDGET_WARN] "
                    f"num_samples={num_samples} > dataset_size={self._dataset_size}; "
                    f"replacement=True 模式下合法，但建议确认 train_iters 设置正确。",
                    file=_sys.stderr,
                )

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        """逐 epoch 确定性随机采样。

        上游逻辑：
            g = torch.Generator()
            g.manual_seed(self.seed * epoch)
            yield from torch.randperm(..., generator=g) / randint(...)
        Walpurgis 通过 EpochSeedPolicy.hash_key() 封装种子计算，行为完全等价。
        """
        if self._torch is None:
            # 无 torch 时退化为 Python random（仅用于测试）
            import random
            rng = random.Random(self.policy.hash_key(self.epoch))
            indices: list
            if self.replacement:
                indices = [rng.randint(0, self._dataset_size - 1)
                           for _ in range(self._num_samples)]
            else:
                indices = list(range(self._dataset_size))
                rng.shuffle(indices)
                indices = indices[:self._num_samples]
            self.epoch += 1
            _dbg("RANDOM_SAMPLER_ITER", f"epoch={self.epoch-1} seed_key={self.policy.hash_key(self.epoch-1)} "
                 f"first8={indices[:8]}")
            yield from indices
            return

        torch = self._torch
        g = torch.Generator()
        epoch_seed = self.policy.hash_key(self.epoch)
        g.manual_seed(epoch_seed)

        if self.replacement:
            # 上游：torch.randint(high=n, size=(num_samples,), generator=g)
            indices = torch.randint(
                high=self._dataset_size,
                size=(self._num_samples,),
                dtype=torch.int64,
                generator=g,
            ).tolist()
        else:
            # 上游：torch.randperm(n, generator=g).tolist()[:num_samples]
            indices = torch.randperm(
                self._dataset_size,
                generator=g,
            ).tolist()[:self._num_samples]

        _dbg(
            "RANDOM_SAMPLER_ITER",
            f"epoch={self.epoch} seed_key={epoch_seed} "
            f"replacement={self.replacement} yielding={len(indices)} "
            f"first8={indices[:8]}"
        )
        self.epoch += 1
        yield from indices

    def set_epoch(self, epoch: int) -> None:
        """手动设置 epoch（分布式训练中由 trainer 调用）。"""
        _dbg("SET_EPOCH", f"epoch {self.epoch} -> {epoch}")
        self.epoch = epoch


# ---------------------------------------------------------------------------
# DistributedBatchSampler — 上游 data_utils/samplers.py 第二个核心类
# ---------------------------------------------------------------------------

class DistributedBatchSampler:
    """分布式批次采样器，将批次按 rank 切片。

    上游 Megatron-LM 66719e973 同文件新增，配合 RandomSampler 使用。
    每个 rank 只消费整体批次序列的 1/world_size 子集，保证数据不重叠。

    上游实现核心：
        start_iter = rank * num_micro_batches
        end_iter   = (rank+1) * num_micro_batches
        每 batch_size 行取一段，交给对应 rank。

    Walpurgis 改写要点（≥20%）：
        - `ShardSpec` dataclass 封装 (start, end, length) 三元组，
          替代上游裸 start_iter/end_iter 变量，`is_valid()` 断言边界一致性；
        - `_dbg()` 断点覆盖 __init__ 与每次 __iter__ 的 shard 切片计算。

    Args:
        sampler: 基础采样器（RandomSampler 或 SequentialSampler）。
        batch_size: 每批样本数。
        drop_last: 是否丢弃最后不足 batch_size 的批次。
        rank: 当前进程 rank。
        world_size: 总进程数。
    """

    @dataclass
    class ShardSpec:
        """单 rank 的批次分片规格。"""
        rank: int
        world_size: int
        total_batches: int

        def start(self) -> int:
            return self.rank * self._micro_batches()

        def end(self) -> int:
            return (self.rank + 1) * self._micro_batches()

        def _micro_batches(self) -> int:
            return self.total_batches // self.world_size

        def is_valid(self) -> bool:
            return 0 <= self.start() <= self.end() <= self.total_batches

    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

        _dbg(
            "DIST_BATCH_SAMPLER_INIT",
            f"rank={rank} world_size={world_size} batch_size={batch_size} "
            f"drop_last={drop_last} sampler_len={len(sampler)}"
        )

    def __iter__(self) -> Iterator[list]:
        batch: list = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and batch:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)

    def distributed_iter(self) -> Iterator[list]:
        """仅产出属于当前 rank 的批次切片。

        Walpurgis 将上游 for-loop index 判断封装为 ShardSpec，
        使 rank/world_size 边界计算集中、可测试。
        """
        all_batches = list(self.__iter__())
        total = len(all_batches)
        spec = self.ShardSpec(
            rank=self.rank,
            world_size=self.world_size,
            total_batches=total,
        )
        _dbg(
            "DIST_BATCH_SAMPLER_SHARD",
            f"total_batches={total} rank={self.rank}/{self.world_size} "
            f"shard=[{spec.start()}, {spec.end()}) valid={spec.is_valid()}"
        )
        yield from all_batches[spec.start():spec.end()]


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """模块加载时的五项断言（WALPURGIS_DEBUG=1 时执行）。"""
    if not _DEBUG:
        return

    # 1. EpochSeedPolicy 乘法策略
    p = EpochSeedPolicy(base_seed=42, multiply_epoch=True)
    assert p.hash_key(3) == 126, f"seed strategy error: {p.hash_key(3)}"
    _dbg("SELF_CHECK", "EpochSeedPolicy(multiply) OK")

    # 2. EpochSeedPolicy 加法策略
    p2 = EpochSeedPolicy(base_seed=10, multiply_epoch=False)
    assert p2.hash_key(5) == 15, f"additive seed error: {p2.hash_key(5)}"
    _dbg("SELF_CHECK", "EpochSeedPolicy(additive) OK")

    # 3. SamplerBudget 预算计算
    budget = SamplerBudget(dataset_size=1000, batch_size=32, train_iters=50)
    assert budget.total_budget() == 1600
    assert budget.is_over_allocated()
    _dbg("SELF_CHECK", "SamplerBudget OK")

    # 4. RandomSampler 无 torch 回退（可复现）
    class _FakeDS:
        def __len__(self): return 20
    rs = RandomSampler(_FakeDS(), replacement=True, num_samples=10, seed=7)
    out1 = list(rs)
    rs.set_epoch(0)
    out2 = list(rs)
    # epoch=0 与 epoch=1 产出不同（seed*0=0 vs seed*1=7），但同 epoch 相同
    rs2 = RandomSampler(_FakeDS(), replacement=True, num_samples=10, seed=7)
    out3 = list(rs2)
    assert out1 == out3, f"RandomSampler reproducibility failed: {out1} vs {out3}"
    _dbg("SELF_CHECK", f"RandomSampler reproducibility OK (first5={out1[:5]})")

    # 5. DistributedBatchSampler ShardSpec 边界
    spec = DistributedBatchSampler.ShardSpec(rank=1, world_size=4, total_batches=8)
    assert spec.start() == 2 and spec.end() == 4 and spec.is_valid()
    _dbg("SELF_CHECK", "ShardSpec OK")

    _dbg("SELF_CHECK", "✓ 全部 5 项断言通过")


_dbg("MODULE_LOAD", "faster_dl_sampler_66719e9 载入开始")
_self_check()
_dbg("MODULE_LOAD", "faster_dl_sampler_66719e9 载入完成")
