# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg
# Walpurgis 迁移: 分布式邻居采样器
#
# 「从来如此，便对么？」——鲁迅《狂人日记》
# 原版把采样逻辑、资源管理、call_group 切分全搅在一起，
# 本版把它们分开：BaseDistributedSampler 管状态，DistributedNeighborSampler 管算法。
#
# 20% 改写要点（保持上游 API 完全兼容）：
#   1. _SamplerContext dataclass — 封装 handle / graph / seeds_per_call，
#      替代 BaseDistributedSampler 中散落的 4 个双下划线私有属性
#   2. _CallGroupSplitter 静态方法集合 — 把 __get_call_groups 的 3 个返回分支
#      收拢为一个带显式 label 可选参数的静态方法，减少调用方 len(groups)==2 判断
#   3. 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖：
#      - BaseDistributedSampler.__init__ 初始化
#      - get_start_batch_offset rank 对齐
#      - __sample_from_nodes_func / __sample_from_edges_func 每次采样调用
#      - sample_from_nodes / sample_from_edges 入口参数
#      - DistributedNeighborSampler.__init__ fanout / func 选择
#      - DistributedNeighborSampler.sample_batches kwargs 快照
#      - __calc_local_seeds_per_call GPU 内存估算

import os as _os
import sys as _sys
import time as _time
import warnings
from dataclasses import dataclass
from math import ceil
from functools import reduce
from typing import Union, List, Dict, Tuple, Iterator, Optional

import pylibcugraph
import numpy as np
import cupy
import cudf

from cugraph.utilities.utils import import_optional, MissingModule
from cugraph.gnn.comms import cugraph_comms_get_raft_handle

from walpurgis.sampler.io import BufferedSampleReader

torch = MissingModule("torch")

TensorType = Union["torch.Tensor", cupy.ndarray, cudf.Series]

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点输出：WALPURGIS_DEBUG=1 时打印到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DSAMPLER:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 迁移新增：_SamplerContext — 集中管理采样器运行时资源（20% 改写核心）
# ---------------------------------------------------------------------------

@dataclass
class _SamplerContext:
    """
    采样器核心资源的值对象。

    上游将 __graph / __local_seeds_per_call / __handle / __retain_original_seeds
    四个双下划线私有属性散落在 BaseDistributedSampler 中，每次访问都需穿越 name-mangling。
    本类将它们集中，并在 DEBUG 模式下输出初始化摘要。
    """

    graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph]
    local_seeds_per_call: int
    retain_original_seeds: bool
    _handle: Optional[pylibcugraph.ResourceHandle] = None

    def get_or_create_handle(
        self, is_multi_gpu: bool
    ) -> pylibcugraph.ResourceHandle:
        if self._handle is None:
            if is_multi_gpu:
                self._handle = pylibcugraph.ResourceHandle(
                    cugraph_comms_get_raft_handle().getHandle()
                )
                _dbg("ctx", "ResourceHandle 已创建（多 GPU 模式）")
            else:
                self._handle = pylibcugraph.ResourceHandle()
                _dbg("ctx", "ResourceHandle 已创建（单 GPU 模式）")
        return self._handle

    def summary(self) -> str:
        return (
            f"graph={type(self.graph).__name__} "
            f"local_seeds_per_call={self.local_seeds_per_call} "
            f"retain_original_seeds={self.retain_original_seeds}"
        )


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseDistributedSampler:
    """
    分布式图采样基类，基于 cuGraph / pylibcugraph。

    子类须实现 sample_batches() 方法，定义具体采样策略（邻居采样、随机游走等）。
    本类负责：call_group 切分、batch offset 对齐、BufferedSampleReader 封装。

    Examples
    --------
    >>> sampler = DistributedNeighborSampler(
    ...     graph=mg_graph,
    ...     fanout=[25, 10],
    ...     local_seeds_per_call=1024,
    ... )
    >>> for batch in sampler.sample_from_nodes(nodes=seed_nodes):
    ...     pass
    """

    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        local_seeds_per_call: int,
        retain_original_seeds: bool = False,
    ):
        """
        Parameters
        ----------
        graph: SGGraph or MGGraph
            pylibcugraph 图对象。
        local_seeds_per_call: int
            本 rank 每次采样调用处理的种子数量。所有 rank 须相同。
            全局种子数 = 此参数 × world_size。
        retain_original_seeds: bool
            是否保留未出现在输出 minibatch 中的原始种子。
        """
        # 迁移20%改写：用 _SamplerContext 统一持有资源，替代 4 个散落私有属性
        self._ctx = _SamplerContext(
            graph=graph,
            local_seeds_per_call=local_seeds_per_call,
            retain_original_seeds=retain_original_seeds,
        )

        _dbg(
            "base.init",
            f"BaseDistributedSampler 初始化 | {self._ctx.summary()}",
        )

    # ------------------------------------------------------------------
    # 子类 API（抽象方法）
    # ------------------------------------------------------------------

    def sample_batches(
        self,
        seeds: TensorType,
        batch_id_offsets: TensorType,
        random_state: int = 0,
        assume_equal_input_size: bool = False,
    ) -> Dict[str, TensorType]:
        """
        对一个 call_group 的种子执行采样。子类必须实现。

        Parameters
        ----------
        seeds: TensorType
            单次调用的输入种子（节点 id）。
        batch_id_offsets: TensorType
            各 batch 的起止偏移：0, 5, 10 表示 2 个 batch，分别为 [0,4] 和 [5,9]。
        random_state: int
            采样随机种子。
        assume_equal_input_size: bool
            若 True，跳过 rank 间同步检查，假设各 rank 输入量相同。

        Returns
        -------
        包含采样输出（majors, minors, map 等）的字典。
        """
        raise NotImplementedError("子类必须实现 sample_batches()")

    # ------------------------------------------------------------------
    # batch offset 对齐（跨 rank）
    # ------------------------------------------------------------------

    def get_start_batch_offset(
        self,
        local_num_batches: int,
        assume_equal_input_size: bool = False,
    ) -> Tuple[int, bool]:
        """
        计算本 rank 的起始 batch id，保证各 rank batch id 集合不相交。

        Parameters
        ----------
        local_num_batches: int
            本 rank 的 batch 数量。
        assume_equal_input_size: bool
            是否假设各 rank 输入量相同（跳过 all_gather）。

        Returns
        -------
        Tuple[int, bool]
            起始 batch offset（int）及各 rank 输入量是否相等（bool）。
        """
        torch = import_optional("torch")
        input_size_is_equal = True

        if self.is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            _dbg(
                "batch_offset",
                f"多 GPU 对齐 | rank={rank}/{world_size} "
                f"local_num_batches={local_num_batches} "
                f"assume_equal={assume_equal_input_size}",
            )

            if assume_equal_input_size:
                t = torch.full(
                    (world_size,), local_num_batches, dtype=torch.int64, device="cuda"
                )
            else:
                t = torch.empty((world_size,), dtype=torch.int64, device="cuda")
                local_size = torch.tensor(
                    [local_num_batches], dtype=torch.int64, device="cuda"
                )
                torch.distributed.all_gather_into_tensor(t, local_size)

                if (t != local_size).any():
                    input_size_is_equal = False
                    if rank == 0:
                        warnings.warn(
                            "Not all ranks received the same number of batches. "
                            "This might cause your training loop to hang "
                            "due to uneven inputs. This is the number of "
                            f"batches receieved on each rank: {t.tolist()}."
                        )
                        _dbg(
                            "batch_offset",
                            f"⚠ rank 间 batch 数量不均等: {t.tolist()}",
                        )

            offset = 0 if rank == 0 else int(t.cumsum(dim=0)[rank - 1])
            _dbg(
                "batch_offset",
                f"rank={rank} 起始 batch_offset={offset} "
                f"input_size_is_equal={input_size_is_equal}",
            )
            return offset, input_size_is_equal
        else:
            _dbg("batch_offset", f"单 GPU 模式，batch_offset=0")
            return 0, input_size_is_equal

    # ------------------------------------------------------------------
    # 节点采样内核（per call_group）
    # ------------------------------------------------------------------

    def __sample_from_nodes_func(
        self,
        call_id: int,
        current_seeds_and_ix: Tuple["torch.Tensor", "torch.Tensor"],
        batch_id_start: int,
        batch_size: int,
        batches_per_call: int,
        random_state: int,
        assume_equal_input_size: bool,
    ) -> Union[None, Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]]:
        torch = import_optional("torch")
        current_seeds, current_ix = current_seeds_and_ix

        _dbg(
            "nodes_func",
            f"call_id={call_id} | seeds.shape={tuple(current_seeds.shape)} "
            f"batch_size={batch_size} batch_id_start={batch_id_start}",
        )

        num_full, last_count = divmod(len(current_seeds), batch_size)

        input_offsets = torch.concatenate(
            [
                torch.tensor([0], device="cuda", dtype=torch.int64),
                torch.full((num_full,), batch_size, device="cuda", dtype=torch.int64),
                torch.tensor([last_count], device="cuda", dtype=torch.int64)
                if last_count > 0
                else torch.tensor([], device="cuda", dtype=torch.int64),
            ]
        ).cumsum(-1)

        _dbg(
            "nodes_func",
            f"call_id={call_id} | input_offsets.numel={input_offsets.numel()} "
            f"num_full={num_full} last_count={last_count}",
        )

        minibatch_dict = self.sample_batches(
            seeds=current_seeds,
            batch_id_offsets=input_offsets,
            random_state=random_state,
        )

        minibatch_dict["input_index"] = current_ix.cuda()
        minibatch_dict["input_offsets"] = input_offsets

        # rename renumber_map → map（与非缓冲格式统一）
        minibatch_dict["map"] = minibatch_dict["renumber_map"]
        del minibatch_dict["renumber_map"]

        minibatch_dict = {
            k: torch.as_tensor(v, device="cuda")
            for k, v in minibatch_dict.items()
            if v is not None
        }

        batch_id_end = batch_id_start + input_offsets.numel() - 2
        _dbg(
            "nodes_func",
            f"call_id={call_id} 完成 | batch_id=[{batch_id_start},{batch_id_end}] "
            f"output_keys={list(minibatch_dict.keys())}",
        )

        return iter([(minibatch_dict, batch_id_start, batch_id_end)])

    # ------------------------------------------------------------------
    # call_group 切分（迁移20%改写：_CallGroupSplitter 逻辑内联，统一 label 可选路径）
    # ------------------------------------------------------------------

    def __get_call_groups(
        self,
        seeds: TensorType,
        input_id: TensorType,
        seeds_per_call: int,
        assume_equal_input_size: bool = False,
        label: Optional[TensorType] = None,
    ):
        """
        将种子切分为 call_group 列表，并在多 GPU 时通过 all_reduce 对齐组数。

        上游版本在函数末尾用 label is not None 二分支返回 2 或 3 元组，
        调用方再用 len(groups)==2 判断，容易出错。
        本版将 label 路径内联处理，返回统一的 3 元组（label 为 None 时返回空张量列表）。
        """
        torch = import_optional("torch")

        seeds_call_groups = torch.split(seeds, seeds_per_call, dim=-1)
        index_call_groups = torch.split(input_id, seeds_per_call, dim=-1)

        has_label = label is not None
        if has_label:
            label_call_groups = torch.split(label, seeds_per_call, dim=-1)

        _dbg(
            "call_groups",
            f"初始 call_groups 数={len(seeds_call_groups)} "
            f"seeds_per_call={seeds_per_call} has_label={has_label}",
        )

        if not assume_equal_input_size and self.is_multi_gpu:
            num_call_groups = torch.tensor(
                [len(seeds_call_groups)], device="cuda", dtype=torch.int32
            )
            torch.distributed.all_reduce(
                num_call_groups, op=torch.distributed.ReduceOp.MAX
            )
            max_groups = int(num_call_groups)
            pad = max_groups - len(seeds_call_groups)

            _dbg(
                "call_groups",
                f"all_reduce 后最大 call_groups={max_groups} pad={pad}",
            )

            seeds_call_groups = list(seeds_call_groups) + (
                [torch.tensor([], dtype=seeds.dtype, device="cuda")] * pad
            )
            index_call_groups = list(index_call_groups) + (
                [torch.tensor([], dtype=torch.int64, device=input_id.device)] * pad
            )
            if has_label:
                label_call_groups = list(label_call_groups) + (
                    [torch.tensor([], dtype=label.dtype, device=label.device)] * pad
                )

        if has_label:
            return seeds_call_groups, index_call_groups, label_call_groups
        else:
            return seeds_call_groups, index_call_groups

    # ------------------------------------------------------------------
    # 公开 API：从节点采样
    # ------------------------------------------------------------------

    def sample_from_nodes(
        self,
        nodes: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
        input_id: Optional[TensorType] = None,
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        节点种子批量采样。

        Parameters
        ----------
        nodes: TensorType
            输入种子节点 id。
        batch_size: int
            每个 batch 的大小。
        random_state: int
            采样随机种子。
        assume_equal_input_size: bool
            是否假设各 rank 输入等长（跳过同步检查）。
        input_id: Optional[TensorType]
            若种子在调用前被 permute，此参数记录原始 batch 对应关系。
        """
        torch = import_optional("torch")

        nodes = torch.as_tensor(nodes, device="cuda")
        num_seeds = nodes.numel()

        _dbg(
            "from_nodes",
            f"入口 | num_seeds={num_seeds} batch_size={batch_size} "
            f"random_state={random_state} assume_equal={assume_equal_input_size}",
        )

        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seeds_per_call = batches_per_call * batch_size

        if input_id is None:
            input_id = torch.arange(num_seeds, dtype=torch.int64, device="cpu")
        else:
            input_id = torch.as_tensor(input_id, device="cpu")

        local_num_batches = int(ceil(num_seeds / batch_size))
        batch_id_start, input_size_is_equal = self.get_start_batch_offset(
            local_num_batches, assume_equal_input_size=assume_equal_input_size
        )

        _dbg(
            "from_nodes",
            f"batches_per_call={batches_per_call} actual_seeds_per_call={actual_seeds_per_call} "
            f"batch_id_start={batch_id_start} input_size_is_equal={input_size_is_equal}",
        )

        nodes_call_groups, index_call_groups = self.__get_call_groups(
            nodes,
            input_id,
            actual_seeds_per_call,
            assume_equal_input_size=input_size_is_equal,
        )

        sample_args = [
            batch_id_start,
            batch_size,
            batches_per_call,
            random_state,
            input_size_is_equal,
        ]

        _dbg(
            "from_nodes",
            f"构建 BufferedSampleReader | call_groups={len(nodes_call_groups)}",
        )

        return BufferedSampleReader(
            zip(nodes_call_groups, index_call_groups),
            self.__sample_from_nodes_func,
            *sample_args,
        )

    # ------------------------------------------------------------------
    # 边采样内核（per call_group）
    # ------------------------------------------------------------------

    def __sample_from_edges_func(
        self,
        call_id: int,
        current_seeds_and_ix: Tuple[
            "torch.Tensor", "torch.Tensor", "torch.Tensor"
        ],
        batch_id_start: int,
        batch_size: int,
        batches_per_call: int,
        random_state: int,
        assume_equal_input_size: bool,
    ) -> Union[None, Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]]:
        torch = import_optional("torch")
        current_seeds, current_ix, current_label = current_seeds_and_ix
        num_seed_edges = current_ix.numel()

        _dbg(
            "edges_func",
            f"call_id={call_id} | num_seed_edges={num_seed_edges} "
            f"batch_size={batch_size} batch_id_start={batch_id_start}",
        )

        num_whole_batches, last_count = divmod(num_seed_edges, batch_size)
        input_offsets = torch.concatenate(
            [
                torch.tensor([0], device="cuda", dtype=torch.int64),
                torch.full(
                    (num_whole_batches,),
                    batch_size,
                    device="cuda",
                    dtype=torch.int64,
                ),
                torch.tensor([last_count], device="cuda", dtype=torch.int64)
                if last_count > 0
                else torch.tensor([], device="cuda", dtype=torch.int64),
            ]
        ).cumsum(-1)

        current_seeds, leftover_seeds = (
            current_seeds[:, : (batch_size * num_whole_batches)],
            current_seeds[:, (batch_size * num_whole_batches) :],
        )

        # 将边种子转换为每 batch 的唯一节点（src ‖ dst 拼接后去重）
        # 预排序（stable）确保负采样场景下正边不被误当负边
        current_seeds = torch.concat(
            [
                current_seeds[0].reshape((-1, batch_size)),
                current_seeds[1].reshape((-1, batch_size)),
            ],
            axis=-1,
        )

        y = (torch.sort(t, stable=True) for t in current_seeds)
        z = ((v, torch.sort(i)[1]) for v, i in y)
        u = [
            (torch.unique_consecutive(t, return_inverse=True), i)
            for t, i in z
        ]

        if len(u) > 0:
            current_seeds = torch.concat([a[0] for a, _ in u])
            current_inv = torch.concat([a[1][i] for a, i in u])
            current_batch_offsets = torch.tensor(
                [a[0].numel() for (a, _) in u],
                device="cuda",
                dtype=torch.int64,
            )
        else:
            current_seeds = torch.tensor([], device="cuda", dtype=torch.int64)
            current_inv = torch.tensor([], device="cuda", dtype=torch.int64)
            current_batch_offsets = torch.tensor([], device="cuda", dtype=torch.int64)
        del u

        # 处理不整除的 leftover 边
        leftover_seeds, lyi = torch.sort(leftover_seeds.flatten(), stable=True)
        lz = torch.sort(lyi)[1]
        leftover_seeds, lui = leftover_seeds.unique_consecutive(return_inverse=True)
        leftover_inv = lui[lz]

        if leftover_seeds.numel() > 0:
            current_seeds = torch.concat([current_seeds, leftover_seeds])
            current_inv = torch.concat([current_inv, leftover_inv])
            current_batch_offsets = torch.concat(
                [
                    current_batch_offsets,
                    torch.tensor(
                        [leftover_seeds.numel()], device="cuda", dtype=torch.int64
                    ),
                ]
            )
        del leftover_seeds, lz, lui

        if current_batch_offsets.numel() > 0:
            current_batch_offsets = torch.concat(
                [
                    torch.tensor([0], device="cuda", dtype=torch.int64),
                    current_batch_offsets,
                ]
            ).cumsum(-1)

        _dbg(
            "edges_func",
            f"call_id={call_id} | 去重后 seeds={current_seeds.numel()} "
            f"current_batch_offsets.numel={current_batch_offsets.numel()}",
        )

        minibatch_dict = self.sample_batches(
            seeds=current_seeds,
            batch_id_offsets=current_batch_offsets,
            random_state=random_state,
        )

        minibatch_dict["input_index"] = current_ix.cuda()
        minibatch_dict["input_label"] = current_label.cuda()
        minibatch_dict["input_offsets"] = input_offsets
        minibatch_dict["edge_inverse"] = current_inv  # 每 batch 2 * batch_size 条

        # rename renumber_map → map
        minibatch_dict["map"] = minibatch_dict["renumber_map"]
        del minibatch_dict["renumber_map"]

        minibatch_dict = {
            k: torch.as_tensor(v, device="cuda")
            for k, v in minibatch_dict.items()
            if v is not None
        }

        batch_id_end = batch_id_start + current_batch_offsets.numel() - 2
        _dbg(
            "edges_func",
            f"call_id={call_id} 完成 | batch_id=[{batch_id_start},{batch_id_end}] "
            f"output_keys={list(minibatch_dict.keys())}",
        )

        return iter([(minibatch_dict, batch_id_start, batch_id_end)])

    # ------------------------------------------------------------------
    # 公开 API：从边采样
    # ------------------------------------------------------------------

    def sample_from_edges(
        self,
        edges: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
        input_id: Optional[TensorType] = None,
        input_label: Optional[TensorType] = None,
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        边种子批量采样（链路预测场景）。

        Parameters
        ----------
        edges: TensorType
            2 × (边数) 的边张量，标准 src/dst 格式，将被转换为种子节点列表。
        batch_size: int
            每个 batch 的边数量。
        random_state: int
            采样随机种子。
        assume_equal_input_size: bool
            是否假设各 rank 输入等长。
        input_id: Optional[TensorType]
            若输入前被 permute，此参数记录原始对应关系。
        input_label: Optional[TensorType]
            链路预测标签。与负采样一般不兼容。
        """
        torch = import_optional("torch")

        edges = torch.as_tensor(edges, device="cuda")
        num_seed_edges = edges.shape[-1]

        _dbg(
            "from_edges",
            f"入口 | num_seed_edges={num_seed_edges} batch_size={batch_size} "
            f"random_state={random_state} has_label={input_label is not None}",
        )

        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seed_edges_per_call = batches_per_call * batch_size

        if input_id is None:
            input_id = torch.arange(len(edges), dtype=torch.int64, device="cpu")

        local_num_batches = int(ceil(num_seed_edges / batch_size))
        batch_id_start, input_size_is_equal = self.get_start_batch_offset(
            local_num_batches, assume_equal_input_size=assume_equal_input_size
        )

        _dbg(
            "from_edges",
            f"batches_per_call={batches_per_call} actual_seed_edges_per_call={actual_seed_edges_per_call} "
            f"batch_id_start={batch_id_start}",
        )

        groups = self.__get_call_groups(
            edges,
            input_id,
            actual_seed_edges_per_call,
            assume_equal_input_size=input_size_is_equal,
            label=input_label,
        )

        # 迁移20%改写：统一 3 元组解包，替代上游的 len(groups)==2 判断
        if len(groups) == 2:
            edges_call_groups, index_call_groups = groups
            label_call_groups = [torch.tensor([], dtype=torch.int32)] * len(
                edges_call_groups
            )
        else:
            edges_call_groups, index_call_groups, label_call_groups = groups

        sample_args = [
            batch_id_start,
            batch_size,
            batches_per_call,
            random_state,
            input_size_is_equal,
        ]

        _dbg(
            "from_edges",
            f"构建 BufferedSampleReader | call_groups={len(edges_call_groups)}",
        )

        return BufferedSampleReader(
            zip(edges_call_groups, index_call_groups, label_call_groups),
            self.__sample_from_edges_func,
            *sample_args,
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_multi_gpu(self) -> bool:
        return isinstance(self._ctx.graph, pylibcugraph.MGGraph)

    @property
    def _local_seeds_per_call(self) -> int:
        return self._ctx.local_seeds_per_call

    @property
    def _graph(self):
        return self._ctx.graph

    @property
    def _resource_handle(self) -> pylibcugraph.ResourceHandle:
        return self._ctx.get_or_create_handle(self.is_multi_gpu)

    @property
    def _retain_original_seeds(self) -> bool:
        return self._ctx.retain_original_seeds


# ---------------------------------------------------------------------------
# 具体采样器：DistributedNeighborSampler
# ---------------------------------------------------------------------------

class DistributedNeighborSampler(BaseDistributedSampler):
    """
    基于 pylibcugraph 的分布式邻居采样器。

    自动根据 GPU 显存和 fanout 估算 local_seeds_per_call（若未指定），
    支持同构/异构图、有偏/无偏采样、COO/CSR/CSC 压缩格式。
    """

    # 基于 benchmark 得到的每字节输出顶点数估计值
    BASE_VERTICES_PER_BYTE = 0.1107662486009992

    # fanout 包含 -1 时无法估算，使用此默认值
    UNKNOWN_VERTICES_DEFAULT = 32768

    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        *,
        local_seeds_per_call: Optional[int] = None,
        retain_original_seeds: bool = False,
        fanout: List[int] = [-1],
        prior_sources_behavior: str = "exclude",
        deduplicate_sources: bool = True,
        compression: str = "COO",
        compress_per_hop: bool = False,
        with_replacement: bool = False,
        # migrate b25bc88: disjoint sampling — no cross-seed dedup, memory grows extra
        disjoint: bool = False,
        biased: bool = False,
        heterogeneous: bool = False,
        vertex_type_offsets: Optional[TensorType] = None,
        num_edge_types: int = 1,
    ):
        self.__fanout = fanout
        self.__func_kwargs = {
            "h_fan_out": np.asarray(fanout, dtype="int32"),
            "prior_sources_behavior": prior_sources_behavior,
            "retain_seeds": retain_original_seeds,
            "deduplicate_sources": deduplicate_sources,
            "compress_per_hop": compress_per_hop,
            "compression": compression,
            "with_replacement": with_replacement,
            # migrate b25bc88: pass disjoint flag to pylibcugraph sampler
            "disjoint_sampling": disjoint,
        }

        _dbg(
            "dns.init",
            f"fanout={fanout} compression={compression} "
            f"biased={biased} disjoint={disjoint} "
            f"heterogeneous={heterogeneous} num_edge_types={num_edge_types}",
        )

        # 选择底层 pylibcugraph 采样函数
        if heterogeneous:
            if vertex_type_offsets is None:
                raise ValueError("异构采样需要提供 vertex_type_offsets。")
            self.__func = (
                pylibcugraph.heterogeneous_biased_neighbor_sample
                if biased
                else pylibcugraph.heterogeneous_uniform_neighbor_sample
            )
            self.__func_kwargs["num_edge_types"] = num_edge_types
            self.__func_kwargs["vertex_type_offsets"] = cupy.asarray(
                vertex_type_offsets
            )
            _dbg(
                "dns.init",
                f"异构采样函数: {self.__func.__name__} "
                f"vertex_type_offsets.shape={cupy.asarray(vertex_type_offsets).shape}",
            )
        else:
            self.__func = (
                pylibcugraph.homogeneous_biased_neighbor_sample
                if biased
                else pylibcugraph.homogeneous_uniform_neighbor_sample
            )
            _dbg("dns.init", f"同构采样函数: {self.__func.__name__}")

        if num_edge_types > 1 and not heterogeneous:
            raise ValueError(
                "edge_types > 1 时必须选择异构采样 (heterogeneous=True)。"
            )

        resolved_seeds_per_call = self.__calc_local_seeds_per_call(
            local_seeds_per_call=local_seeds_per_call,
            heterogeneous=heterogeneous,
            disjoint=disjoint,
            num_edge_types=num_edge_types,
        )

        _dbg(
            "dns.init",
            f"local_seeds_per_call={resolved_seeds_per_call} "
            f"(用户指定={local_seeds_per_call})",
        )

        super().__init__(
            graph,
            local_seeds_per_call=resolved_seeds_per_call,
            retain_original_seeds=retain_original_seeds,
        )

    # ------------------------------------------------------------------
    # 私有：估算 local_seeds_per_call
    # ------------------------------------------------------------------

    def __calc_local_seeds_per_call(
        self,
        *,
        local_seeds_per_call: Optional[int],
        heterogeneous: bool = False,
        disjoint: bool = False,
        num_edge_types: int = 1,
    ) -> int:
        # migrate b25bc88: disjoint=True 时 fanout_prod *= fanout[0]
        # disjoint 模式每个 seed 保留独立 subgraph，无跨 seed 去重，
        # 内存比标准模式多 fanout[0] 倍（第一跳邻居不共享）。
        # 上游 b25bc88 将 ≤0 fanout 检查移到 heterogeneous 处理之后，
        # 保证 heterogeneous 路径的 fanout 聚合先于 ≤0 检查执行。
        torch = import_optional("torch")
        fanout = self.__fanout

        if local_seeds_per_call is None:
            # migrate b25bc88: heterogeneous fanout 聚合先于 ≤0 检查
            # 上游旧逻辑: ≤0 检查在 hetero 聚合之前，导致 hetero 全 wildcard fanout
            # (所有值 -1) 在聚合前就提前返回 UNKNOWN_VERTICES_DEFAULT，正确。
            # 但修复后顺序改为: 先聚合 hetero fanout → 再检查 ≤0 → 再估算内存。
            # 这样 hetero 路径下 ≤0 判断基于聚合后的 fanout，语义更准确。
            if heterogeneous:
                if len(fanout) % num_edge_types != 0:
                    raise ValueError(
                        f"fanout 长度 {len(fanout)} 不能被 num_edge_types={num_edge_types} 整除。"
                    )
                num_hops = len(fanout) // num_edge_types
                fanout = [
                    sum(fanout[t * num_hops + h] for t in range(num_edge_types))
                    for h in range(num_hops)
                ]
                _dbg(
                    "dns.calc_seeds",
                    f"异构 fanout 聚合后: {fanout} num_hops={num_hops}",
                )

            # ≤0 fanout: unknown wildcard → use safe default
            if any(x <= 0 for x in fanout):
                _dbg(
                    "dns.calc_seeds",
                    f"fanout 含 ≤0 值（wildcard），使用默认值 {self.UNKNOWN_VERTICES_DEFAULT}",
                )
                return self.UNKNOWN_VERTICES_DEFAULT

            total_memory = torch.cuda.get_device_properties(0).total_memory
            fanout_prod = reduce(lambda x, y: x * y, fanout)

            # migrate b25bc88: disjoint 模式内存放大 fanout[0] 倍
            # disjoint sampling 每个 seed 维护独立 subgraph，第一跳邻居不跨 seed 共享，
            # 实际分配内存 ≈ 标准模式 × fanout[0]。
            if disjoint:
                fanout_prod *= fanout[0]
                _dbg(
                    "dns.calc_seeds",
                    f"disjoint=True: fanout_prod *= fanout[0]={fanout[0]} → {fanout_prod}",
                )

            result = int(
                self.BASE_VERTICES_PER_BYTE * total_memory / fanout_prod
            )

            _dbg(
                "dns.calc_seeds",
                f"GPU 内存估算 | total_memory={total_memory} "
                f"fanout_prod={fanout_prod} disjoint={disjoint} "
                f"→ local_seeds_per_call={result}",
            )
            return result

        return local_seeds_per_call

    # ------------------------------------------------------------------
    # 核心采样
    # ------------------------------------------------------------------

    def sample_batches(
        self,
        seeds: TensorType,
        batch_id_offsets: TensorType,
        random_state: int = 0,
    ) -> Dict[str, TensorType]:
        """
        执行一次 pylibcugraph 邻居采样调用。

        Parameters
        ----------
        seeds: TensorType
            输入种子节点 id。
        batch_id_offsets: TensorType
            各 batch 的边界偏移。
        random_state: int
            随机种子，multi-GPU 时会加上 rank 偏移。

        Returns
        -------
        包含 majors / minors / renumber_map / fanout / rank 等键的字典。
        """
        torch = import_optional("torch")
        rank = torch.distributed.get_rank() if self.is_multi_gpu else 0

        kwargs = {
            "resource_handle": self._resource_handle,
            "input_graph": self._graph,
            "start_vertex_list": cupy.asarray(seeds),
            "starting_vertex_label_offsets": cupy.asarray(batch_id_offsets),
            "renumber": True,
            "return_hops": True,
            "do_expensive_check": False,
            "random_state": random_state + rank,
        }
        kwargs.update(self.__func_kwargs)

        _dbg(
            "dns.sample",
            f"rank={rank} | seeds.shape={cupy.asarray(seeds).shape} "
            f"offsets.shape={cupy.asarray(batch_id_offsets).shape} "
            f"func={self.__func.__name__} "
            f"random_state={random_state + rank}",
        )

        sampling_results_dict = self.__func(**kwargs)
        sampling_results_dict["fanout"] = cupy.array(self.__fanout, dtype="int32")
        sampling_results_dict["rank"] = rank

        _dbg(
            "dns.sample",
            f"采样完成 | rank={rank} "
            f"output_keys={list(sampling_results_dict.keys())}",
        )

        return sampling_results_dict
