# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg
# migrate 47e64e8: [BUG] Fix optional dependencies — 删除 DistSampleReader；cudf/torch 改为 import_optional
# Walpurgis 迁移: BufferedSampleReader — 分布式采样缓冲迭代器
#
# 鲁迅曾言：「上等人做稳了奴隶，不必多想；下等人求做奴隶而不得，才须反抗。」
# 原版的 BufferedSampleReader 连名字都不解释自己——本版加了全链路断点。
#
# 47e64e8 改写要点：
#   - DistSampleReader 整个类删除（上游确认该类从 cugraph 迁移时误带入，
#     依赖 cudf.read_parquet + torch.distributed，不属于 BufferedSampleReader 模块）
#   - `torch = MissingModule("torch")` → `torch = import_optional("torch")`
#   - 新增 `cudf = import_optional("cudf")`
#   - BufferedSampleReader.__init__ 中删除 `global torch; torch = import_optional("torch")`
#
# 20% 改写要点（原有，保留）：
#   - 新增 _WalpurgisReaderStats dataclass，追踪已消费 batch 数 / call 组数 / 空 call 数
#   - __next__ 拆出 _advance_reader() 私有方法，责任更清晰
#   - 全链路 WALPURGIS_DEBUG=1 断点 print：初始化 / 每次切换 call_group / 每次 StopIteration

import os as _os
import sys as _sys
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Tuple, Dict

from cugraph.utilities.utils import import_optional

# 47e64e8: torch/cudf 均为可选依赖；DistSampleReader（依赖 cudf.read_parquet）
# 已在上游 47e64e8 删除（该类是从 cugraph 迁移时误带入的）。
# BufferedSampleReader 保留，去除顶层 MissingModule 占位，改为 import_optional。
torch = import_optional("torch")
cudf = import_optional("cudf")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳与 tag。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-READER:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


@dataclass
class _WalpurgisReaderStats:
    """读取器运行时统计：迁移新增，便于 DEBUG 追踪流量。"""

    batches_consumed: int = 0
    call_groups_advanced: int = 0
    empty_calls_skipped: int = 0
    total_tensor_keys: field(default_factory=list) = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.total_tensor_keys is None:
            self.total_tensor_keys = []

    def record_batch(self, tensor_dict: Dict) -> None:
        self.batches_consumed += 1
        if not self.total_tensor_keys:
            self.total_tensor_keys = list(tensor_dict.keys())


class BufferedSampleReader:
    """
    分布式采样结果的惰性缓冲迭代器。

    上游 (cugraph-gnn 03292cf) 的原始版本将全部采样结果放进一个扁平 iter——
    本版保持语义不变，但新增运行时统计 (_WalpurgisReaderStats) 和断点输出，
    方便调试多卡采样时 call_group 分配是否均衡。

    迭代协议：每次 __next__ 返回 (minibatch_dict, batch_id_start, batch_id_end)。
    """

    def __init__(
        self,
        nodes_call_groups: list,  # list[tuple[Tensor, ...]]
        sample_fn: Callable[..., Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]],
        *args,
        **kwargs,
    ):

        self.__sample_args = args
        self.__sample_kwargs = kwargs

        self.__nodes_call_groups = iter(nodes_call_groups)
        self.__sample_fn = sample_fn
        self.__current_call_id = 0
        self.__current_reader: Iterator | None = None

        # 迁移新增：运行时统计对象
        self._stats = _WalpurgisReaderStats()

        _dbg(
            "init",
            f"BufferedSampleReader 初始化完毕 | "
            f"sample_fn={sample_fn.__name__} "
            f"extra_args={len(args)} extra_kwargs={list(kwargs.keys())}",
        )

    # ------------------------------------------------------------------
    # 内部辅助：切换到下一个 call_group 的 reader（迁移20%改写：拆出此方法）
    # ------------------------------------------------------------------
    def _advance_reader(self) -> None:
        """
        推进到下一个 call_group，更新 __current_reader。
        若 call_groups 耗尽，透传 StopIteration。
        """
        next_group = next(self.__nodes_call_groups)  # 可能抛 StopIteration

        _dbg(
            "advance",
            f"切换 call_group | call_id={self.__current_call_id} "
            f"group_type={type(next_group).__name__}",
        )

        self.__current_reader = self.__sample_fn(
            self.__current_call_id,
            next_group,
            *self.__sample_args,
            **self.__sample_kwargs,
        )
        self.__current_call_id += 1
        self._stats.call_groups_advanced += 1

        _dbg(
            "advance",
            f"新 reader 已就绪 | 累计 call_groups={self._stats.call_groups_advanced}",
        )

    # ------------------------------------------------------------------
    # 迭代器协议
    # ------------------------------------------------------------------
    def __next__(self) -> Tuple[Dict[str, "torch.Tensor"], int, int]:
        if self.__current_reader is None:
            # 首次调用：推进到第一个 call_group
            _dbg("next", "首次迭代，推进第一个 call_group")
            self._advance_reader()
        else:
            try:
                out = next(self.__current_reader)
                self._stats.record_batch(out[0])
                _dbg(
                    "next",
                    f"batch 消费 | batch_id=[{out[1]},{out[2]}] "
                    f"累计={self._stats.batches_consumed} "
                    f"keys={list(out[0].keys())}",
                )
                return out
            except StopIteration:
                _dbg(
                    "next",
                    f"call_group {self.__current_call_id - 1} 耗尽，推进下一个",
                )
                self._advance_reader()

        out = next(self.__current_reader)
        self._stats.record_batch(out[0])
        _dbg(
            "next",
            f"batch 消费（推进后首条）| batch_id=[{out[1]},{out[2]}] "
            f"累计={self._stats.batches_consumed} "
            f"keys={list(out[0].keys())}",
        )
        return out

    def __iter__(self) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        _dbg("iter", "__iter__ 调用，返回 self")
        return self
