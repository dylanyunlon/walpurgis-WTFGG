# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 539d0ad: tensor/dist_matrix.py
# 鲁迅拿法20%改写笔记:
#   上游 DistMatrix 是一个沉默的配角——两列 (col/row) 拼成一张图的边。
#   原文最大的问题: local_col / local_row 的分片计算逻辑完全没有调试出口,
#   一旦 world_size 和 sz 不整除, 极难判断哪个 rank 拿到了哪些边。
#   我们的20%改写:
#     1. __init__ 打印 format/backend/col-row shape;
#     2. local_col / local_row 打印本 rank 的 ix 范围;
#     3. __setitem__ 校验 COO 格式下的 idx/val 兼容性后加断点;
#     4. local_coo 打印聚合后的 shape;
#     5. 把 local_col/local_row 中重复的 arange 逻辑提取为 _local_range(),
#        减少代码重复 (鲁迅最厌重复)。

import os
import sys
import time
from typing import Optional, Union, Tuple, List, Literal

from walpurgis.utils.imports import import_optional
from walpurgis.tensor import DistTensor

torch = import_optional("torch")

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印 (WALPURGIS_DEBUG=1)"""
    if _DEBUG:
        print(
            f"[WALPURGIS-TENSOR:DistMatrix][{time.strftime('%H:%M:%S')}][{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


def _local_range(sz: int, world_size: int, rank: int) -> "torch.Tensor":
    """计算 rank 在 sz 个元素上的本地 range 索引 (range-partition)。

    鲁迅: 原文 local_col / local_row 里各有一份相同的分片公式,
    看上去像照镜子——提取成函数是最小的尊重。
    """
    q, r = divmod(sz, world_size)
    if rank < r:
        start = q * rank + rank
        end = q * (rank + 1) + rank + 1
    else:
        start = q * rank + r
        end = q * (rank + 1) + r

    _dbg(
        "_local_range",
        f"sz={sz} world_size={world_size} rank={rank} "
        f"→ [{start}, {end}) count={end - start}",
    )
    return torch.arange(start, end)


class DistMatrix:
    """WholeGraph 分布式矩阵接口 (Walpurgis 版)。

    以 COO 或 CSC 格式存储分布式稀疏矩阵的 col/row 索引。
    当前 COO 格式支持读写; CSC 格式尚未完整实现 (与上游一致)。

    鲁迅: 矩阵是关系的沉淀——每一条边都是两个节点的因缘。
    Walpurgis 的边表不像冷冰冰的地址簿; 它知道自己身处哪个 rank,
    也知道自己负责图的哪一段。
    """

    def __init__(
        self,
        src: Optional[
            Union[
                Tuple["torch.Tensor", "torch.Tensor"],
                Tuple[DistTensor, DistTensor],
                str,
                List[str],
            ]
        ] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        backend: Optional[Literal["nccl", "vmm"]] = "nccl",
        format: Optional[Literal["csc", "coo"]] = "coo",
    ):
        self.__backend = backend
        self._format = format

        _dbg(
            "__init__",
            f"format={format} backend={backend} device={device} "
            f"shape={shape} dtype={dtype} src_type={type(src).__name__}",
        )

        if isinstance(src, (tuple, list)):
            # list 首元素若为 str 则是文件列表 (未实现)
            if isinstance(src[0], str):
                raise NotImplementedError(
                    "从文件或文件列表构造 DistMatrix 尚未支持。"
                )

            if len(src) != 2:
                raise ValueError("src 必须是包含两个张量的 tuple/list: (col, row)。")

            col_src, row_src = src[0], src[1]
            self._col = DistTensor(
                src=col_src, device=device, dtype=(dtype or col_src.dtype)
            )
            self._row = DistTensor(
                src=row_src, device=device, dtype=(dtype or row_src.dtype)
            )

            _dbg(
                "__init__",
                f"col.shape={self._col.shape} row.shape={self._row.shape}",
            )

            if self._format == "coo":
                if self._col.shape[0] != self._row.shape[0]:
                    raise ValueError(
                        f"COO 格式下 col 和 row 的元素数必须相同: "
                        f"col={self._col.shape[0]}, row={self._row.shape[0]}。"
                    )

        elif src is None:
            if dtype is None or shape is None:
                raise ValueError(
                    "src=None 时必须同时提供 shape 和 dtype。"
                )
            if self._format != "coo":
                raise ValueError(
                    "空矩阵仅支持 COO 格式 (format='coo')。"
                )

            self._col = DistTensor(
                src=None, device=device, dtype=dtype,
                shape=(shape[0],), backend=self.__backend,
            )
            self._row = DistTensor(
                src=None, device=device, dtype=dtype,
                shape=(shape[1],), backend=self.__backend,
            )
            _dbg("__init__", f"空矩阵创建完毕 shape={shape}")

        elif isinstance(src, str):
            raise NotImplementedError(
                "从单一文件构造 DistMatrix 尚未支持。"
            )
        else:
            raise ValueError(
                f"无效的 src 类型: {type(src).__name__}。"
                "支持: (col_tensor, row_tensor) tuple 或 None。"
            )

    # ── 写 ────────────────────────────────────────────────────────────────────

    def __setitem__(
        self,
        idx: Union["torch.Tensor", slice],
        val: Union["torch.Tensor", "tuple[torch.Tensor, torch.Tensor]"],
    ):
        """按索引更新 COO 边 (col, row)。

        val 可以是:
          - 2×N torch.Tensor: val[0]=col, val[1]=row
          - (col_tensor, row_tensor) tuple
        """
        if isinstance(idx, slice):
            size = self._col.shape[0]
            idx = torch.arange(size)[idx]

        if self._format != "coo":
            raise ValueError("目前只支持 COO 格式的更新操作。")

        _dbg(
            "__setitem__",
            f"idx.shape={list(idx.shape)} val_type={type(val).__name__}",
        )

        if isinstance(val, torch.Tensor):
            if val.dim() != 2:
                raise ValueError(f"val 必须是 2D Tensor, 实际 dim={val.dim()}。")
            if val.shape[0] != 2:
                raise ValueError(
                    f"val 必须是 2×N Tensor, 实际 shape={list(val.shape)}。"
                )
            if val.shape[1] != idx.shape[0]:
                raise ValueError(
                    f"val.shape[1]={val.shape[1]} 与 idx.shape[0]={idx.shape[0]} 不匹配。"
                )
            self._col[idx] = val[0]
            self._row[idx] = val[1]

        elif isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError(
                    f"val tuple 必须含恰好两个张量, 实际 len={len(val)}。"
                )
            self._col[idx] = val[0]
            self._row[idx] = val[1]
        else:
            raise TypeError(
                f"不支持的 val 类型: {type(val).__name__}。"
                "期望 torch.Tensor (2×N) 或 (col, row) tuple。"
            )

        _dbg("__setitem__", "写入完毕 ✓")

    # ── 读 ────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: "torch.Tensor") -> "torch.Tensor":
        """按全局索引读取 COO 边, 返回 2×len(idx) Tensor。"""
        if self._format != "coo":
            raise ValueError("目前只支持 COO 格式的读取操作。")
        if idx.dim() != 1:
            raise ValueError(f"idx 必须是 1D Tensor, 实际 dim={idx.dim()}。")

        _dbg("__getitem__", f"idx.shape={list(idx.shape)}")
        result = torch.stack([self._col[idx], self._row[idx]])
        _dbg("__getitem__", f"result.shape={list(result.shape)}")
        return result

    def get_local_tensor(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """返回本 rank 的本地 (col, row) 张量 tuple。"""
        return (self._col.get_local_tensor(), self._row.get_local_tensor())

    # ── 本地分片属性 ──────────────────────────────────────────────────────────

    @property
    def local_col(self) -> "torch.Tensor":
        """本 rank 负责的 col 分片 (range-partition)。"""
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        ix = _local_range(self._col.shape[0], world_size, rank)
        return self._col[ix]

    @property
    def local_row(self) -> "torch.Tensor":
        """本 rank 负责的 row 分片 (range-partition)。"""
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        ix = _local_range(self._row.shape[0], world_size, rank)
        return self._row[ix]

    @property
    def local_coo(self) -> "torch.Tensor":
        """本 rank 的 COO 边矩阵 (2×local_N)。"""
        coo = torch.stack([self.local_col, self.local_row])
        _dbg("local_coo", f"shape={list(coo.shape)}")
        return coo

    # ── 元信息属性 ────────────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, int]:
        """(col_size, row_size): 即 (边数, 边数) for COO。"""
        return (self._col.shape[0], self._row.shape[0])

    @property
    def dtype(self) -> "torch.dtype":
        return self._col.dtype
