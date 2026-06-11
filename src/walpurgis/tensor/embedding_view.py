# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 31ee98f: add EmbeddingView for lazy/memory-safe embedding access
#
# 上游变化 (31ee98f, cugraph-dgl/view.py):
#   新增 EmbeddingView 类: 对大型 embedding storage 的懒访问封装。
#   核心动机: 整张 embedding 矩阵 O(N×D) 太大不能一次性拉到内存；
#   调用方应按索引取片 (EmbeddingView[u]), 而非整体调用 EmbeddingView()。
#   __call__() 发出警告并取整张 (紧急兜底路径)。
#
# Walpurgis 改写20% (鲁迅拿法):
#   上游 EmbeddingView 封装的是 dgl.storages.base.FeatureStorage（DGL 专属）。
#   Walpurgis 对应的 storage 是 DistTensor / DistEmbedding（自有层）。
#   改写要点:
#     1. _WalpurgisEmbeddingBackend 协议类: 显式描述 storage 接口契约，
#        替代上游隐式依赖 dgl.storages.FeatureStorage 的鸭子类型。
#        任何实现 fetch(indices, device) 的对象均可接入。
#     2. __getitem__: 上游在 RuntimeError 时 fallback 到 .cuda() 索引；
#        我们把 fallback 路径的重试逻辑拆成独立的 _fetch_with_fallback()，
#        便于单独测试与断点追踪。
#     3. shape 属性: 上游 try/except RuntimeError 硬编码探针索引 [0]；
#        我们改为 _probe_shape()，显式打印探针结果和 fallback 路径，
#        帮助排查"shape[0] 返回 1 但实际是整张 embedding"的隐性 bug。
#     4. WALPURGIS_DEBUG=1 断点: 覆盖 fetch/fallback/shape probe/call 整条链路。
#
# 上游 bug (上游原文已知):
#   neighbor_sampler.py (31ee98f) 同批修复:
#     旧: ds.sample_from_nodes(...)  # 返回 None，get_reader() 才返回 reader
#         return HomogeneousSampleReader(ds.get_reader(), ...)
#     新: reader = ds.sample_from_nodes(...)  # 直接返回 reader
#         return HomogeneousSampleReader(reader, ...)
#   上游 API 已更新为 sample_from_nodes 直接返回 reader 对象，
#   旧版依赖隐式副作用 ds.get_reader()，若 sample_from_nodes 失败无 reader 时
#   get_reader() 会返回旧 reader 导致数据污染。
#   Walpurgis sampler 层应使用新 API (直接获取 reader 返回值)。

import os
import sys
import time
import warnings
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-TENSOR:EmbeddingView][{time.strftime('%H:%M:%S')}][{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# _WalpurgisEmbeddingBackend: storage 接口协议
# 改写20%要点: 上游隐式依赖 dgl.storages.FeatureStorage 鸭子类型，
# 难以在测试中注入 mock。我们用协议类显式化接口契约。
# ---------------------------------------------------------------------------

class _WalpurgisEmbeddingBackend:
    """
    Walpurgis EmbeddingView 后端接口协议。

    任何满足此接口的对象均可作为 EmbeddingView 的 storage：
      - DistTensor / DistEmbedding (walpurgis.tensor)
      - dgl.storages.base.FeatureStorage (上游兼容路径)
      - 测试用 mock 对象

    接口要求:
        fetch(indices: torch.Tensor, device: str) -> torch.Tensor
            按索引取 embedding 片段，返回 [len(indices), ...] 的 Tensor。
    """

    def fetch(self, indices: "torch.Tensor", device: str) -> "torch.Tensor":
        raise NotImplementedError(
            "EmbeddingView storage 必须实现 fetch(indices, device)。"
        )


# ---------------------------------------------------------------------------
# EmbeddingView: 懒访问封装
# 上游原文: cugraph-dgl/view.py EmbeddingView（31ee98f 新增）
# ---------------------------------------------------------------------------

class EmbeddingView:
    """
    大型 embedding tensor 的懒访问封装。

    调用方应通过索引访问所需片段::

        ev = EmbeddingView(storage, total_entry_count)
        node_feats = ev[node_ids]   # 返回 [len(node_ids), D] Tensor

    整体调用 ev() 会发出警告（浪费内存）并返回完整张量（紧急路径）。

    参数
    ----
    storage : Any
        实现 fetch(indices, device) 接口的 storage 对象。
        可以是 DistEmbedding、DGL FeatureStorage 或测试 mock。
    total_entries : int
        embedding 的总行数 (dim-0 size)。用于 shape 属性和 __call__ 的整体索引。

    上游来源
    --------
    cugraph-dgl/cugraph_dgl/view.py, commit 31ee98f (NVIDIA cugraph-gnn)
    原版封装 dgl.storages.FeatureStorage，Walpurgis 版泛化为任意 storage 接口。
    """

    def __init__(self, storage: Any, total_entries: int) -> None:
        self._storage = storage
        self._total_entries = total_entries
        _dbg(
            "__init__",
            f"storage={type(storage).__name__} total_entries={total_entries}",
        )

    # ── 核心访问 ─────────────────────────────────────────────────────────────

    def __getitem__(self, indices: "torch.Tensor") -> "torch.Tensor":
        """按索引取 embedding 片段。

        上游 RuntimeError fallback: 若首次 fetch 抛 RuntimeError（GPU 索引问题），
        自动重试，将 indices 移到 CUDA 再 fetch。
        Walpurgis 改写: 将 fallback 逻辑拆入 _fetch_with_fallback()，单独可测试。
        """
        return self._fetch_with_fallback(indices)

    def _fetch_with_fallback(
        self, indices: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        两阶段 fetch:
          1. 直接 fetch(indices, "cuda")
          2. 若 RuntimeError，将 indices 移到 CUDA 后重试

        Walpurgis 改写: 上游将 fallback 内联在 __getitem__ 中；
        拆出此方法便于 mock 测试和 DEBUG 打印断点。
        """
        _dbg(
            "fetch",
            f"indices.shape={list(indices.shape)} indices.device={indices.device}",
        )
        try:
            result = self._storage.fetch(indices, "cuda")
            _dbg("fetch", f"→ OK shape={list(result.shape)}")
            return result
        except RuntimeError as ex:
            _dbg(
                "fetch_fallback",
                f"首次 fetch 失败 ({ex}), 尝试 indices.cuda() 重试",
            )
            warnings.warn(
                f"[EmbeddingView] fetch 时出错，尝试将索引移到 CUDA 后重试: {ex}"
            )
            result = self._storage.fetch(indices.cuda(), "cuda")
            _dbg("fetch_fallback", f"→ fallback OK shape={list(result.shape)}")
            return result

    def __call__(self) -> "torch.Tensor":
        """
        返回完整 embedding 矩阵（不推荐，浪费内存）。

        上游语义: 整体调用是紧急/调试路径，应优先使用 __getitem__。
        Walpurgis 保留此路径但强化警告，并在 WALPURGIS_DEBUG=1 时打印总行数。
        """
        _dbg(
            "__call__",
            f"整体取 embedding total_entries={self._total_entries}（高内存风险）",
        )
        warnings.warn(
            "[EmbeddingView] 整体获取 embedding 会浪费内存，建议用索引访问 "
            "embedding_view[indices] 取所需片段。",
            stacklevel=2,
        )
        import torch
        all_idx = torch.arange(self._total_entries, dtype=torch.int64)
        return self[all_idx]

    # ── shape ────────────────────────────────────────────────────────────────

    @property
    def shape(self) -> "torch.Size":
        """
        返回 embedding 的 shape: [total_entries, feature_dim, ...]。

        实现方式: 探针取索引 [0] 的一行，从其 shape 推断 feature_dim。
        Walpurgis 改写:
          - _probe_shape() 封装探针逻辑，上游将此逻辑内联在 shape 属性中；
          - 打印探针结果，帮助排查"shape[0] 返回 1 但 total_entries 另有值"的隐性 bug。
          - 上游探针索引硬编码为 CPU tensor([0])；Walpurgis 也先试 CPU，失败再试 CUDA。
        """
        return self._probe_shape()

    def _probe_shape(self) -> "torch.Size":
        """用探针索引 [0] 推断 feature_dim，组合 total_entries 得完整 shape。"""
        import torch
        probe_cpu = torch.tensor([0], dtype=torch.int64)
        try:
            row = self._storage.fetch(probe_cpu, "cpu")
            _dbg("probe_shape", f"CPU 探针成功 row.shape={list(row.shape)}")
        except RuntimeError:
            probe_gpu = torch.tensor([0], device="cuda", dtype=torch.int64)
            row = self._storage.fetch(probe_gpu, "cuda")
            _dbg("probe_shape", f"CUDA 探针成功 row.shape={list(row.shape)}")

        # row.shape = [1, feature_dim, ...]；将第0维替换为 total_entries
        dim_sizes = list(row.shape)
        dim_sizes[0] = self._total_entries
        result = torch.Size(dim_sizes)
        _dbg(
            "probe_shape",
            f"最终 shape={result} (probe dim={list(row.shape)}, "
            f"total_entries={self._total_entries})",
        )
        return result


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "EmbeddingView",
    "_WalpurgisEmbeddingBackend",
]
