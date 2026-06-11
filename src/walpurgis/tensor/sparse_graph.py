# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate db74d87: Merge pull request #2 from alexbarghi-nv/copy-from-cugraph
# migrate 1295d2f: update branch — PEP 8 E225: f-string 内算术表达式加空格
#                  size[0]+1 → size[0] + 1，size[1]+1 → size[1] + 1
# 上游源: python/cugraph-dgl/cugraph_dgl/nn/conv/base.py (SparseGraph)
# Walpurgis 迁移: 稀疏图多格式转换工具
#
# 鲁迅《准风月谈·文人无文》:
# 「中国人本来是不敢反抗的，连抱怨的勇气也消失了。」
# 上游 SparseGraph 把 COO/CSC/CSR 格式切换逻辑拆得极散——
# 谁要用谁自己看代码，没有调试入口，出错只能靠猜。
# 本版把责任边界重新画清楚，加全链路断点，让每次格式转换有迹可循。
#
# 20% 改写要点（保持上游 API 完全兼容）：
#   1. _validate_inputs() 私有方法 — 把原 __init__ 中散落的 6 处 if/raise 集中,
#      同时在 DEBUG 模式下打印输入摘要（dtype/shape/设备）
#   2. _build_csc() 私有方法 — 把"首次建 CSC"的 sort+compress 路径独立出来,
#      原版内联在 __init__ 中不易单测
#   3. 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖：
#      - __init__ 参数校验完成后（打印 size/nnz/formats/is_sorted）
#      - _build_csc 排序/压缩时（打印耗时 placeholder 和 cdst_ids.shape）
#      - src_ids / dst_ids / csrc_ids / cdst_ids 各 lazy 属性首次计算时
#      - reduce_memory 执行时（打印被 None 掉的张量名列表）

import os as _os
import sys as _sys
import time as _time
from typing import Optional, Tuple, Union

from walpurgis.utils.imports import import_optional

torch = import_optional("torch")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-SPARSE:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 辅助函数（上游 base.py 顶层函数，保持原名）
# ---------------------------------------------------------------------------

def compress_ids(ids: "torch.Tensor", size: int) -> "torch.Tensor":
    """将 COO 行索引（已排序）压缩为 CSR/CSC 的指针数组。

    上游实现等价于 torch._convert_indices_from_coo_to_csr，此处保留原接口。
    """
    return torch._convert_indices_from_coo_to_csr(
        ids, size, out_int32=ids.dtype == torch.int32
    )


def decompress_ids(c_ids: "torch.Tensor") -> "torch.Tensor":
    """将 CSR/CSC 指针数组还原为 COO 行索引（等价于 repeat_interleave）。"""
    ids = torch.arange(c_ids.numel() - 1, dtype=c_ids.dtype, device=c_ids.device)
    return ids.repeat_interleave(c_ids[1:] - c_ids[:-1])


# ---------------------------------------------------------------------------
# SparseGraph
# ---------------------------------------------------------------------------

class SparseGraph:
    r"""稀疏图的多格式表示：始终持有 CSC，按需生成 COO / CSR。

    迁移自 cugraph-dgl ``cugraph_dgl.nn.conv.base.SparseGraph``（db74d87）。
    在 Walpurgis 中用于表示时空图的局部邻接结构，替代直接传 DGLHeteroGraph 给
    pylibcugraphops 算子的模式。

    Parameters
    ----------
    size: tuple of int
        邻接矩阵尺寸 (num_src_nodes, num_dst_nodes)。
    src_ids: torch.Tensor
        边的源节点索引（COO 格式）。
    dst_ids: torch.Tensor, optional
        边的目的节点索引。``dst_ids`` 与 ``cdst_ids`` 二者必须提供其一。
    csrc_ids: torch.Tensor, optional
        CSR 压缩源指针，shape ``(num_src_nodes + 1,)``。
    cdst_ids: torch.Tensor, optional
        CSC 压缩目的指针，shape ``(num_dst_nodes + 1,)``。
    values: torch.Tensor, optional
        边的权重或边类型。
    is_sorted: bool
        若 COO 输入已按 dst 升序排列，设为 True 可跳过排序，加速 CSC 建立。
    formats: str or tuple of str
        期望预建的格式集合，必须包含 ``"csc"``。默认 ``"csc"``。
    reduce_memory: bool
        构建完毕后释放不需要的中间张量。默认 True。

    Notes
    -----
    MFG（Message Flow Graph）场景下节点 id 必须已完成 renumber。
    """

    # 每种格式需要的张量名（属性名）
    supported_formats = {
        "coo": ("_src_ids", "_dst_ids"),
        "csc": ("_cdst_ids", "_src_ids"),
        "csr": ("_csrc_ids", "_dst_ids", "_perm_csc2csr"),
    }

    # 所有可能持有的张量名集合
    all_tensors = frozenset([
        "_src_ids",
        "_dst_ids",
        "_csrc_ids",
        "_cdst_ids",
        "_perm_coo2csc",
        "_perm_csc2csr",
    ])

    def __init__(
        self,
        size: Tuple[int, int],
        src_ids: "torch.Tensor",
        dst_ids: Optional["torch.Tensor"] = None,
        csrc_ids: Optional["torch.Tensor"] = None,
        cdst_ids: Optional["torch.Tensor"] = None,
        values: Optional["torch.Tensor"] = None,
        is_sorted: bool = False,
        formats: Union[str, Tuple[str, ...]] = "csc",
        reduce_memory: bool = True,
    ):
        self._num_src_nodes, self._num_dst_nodes = size
        self._is_sorted = is_sorted

        # 20% 改写：集中校验，替代原版 6 处散落 if/raise
        self._validate_inputs(
            size, src_ids, dst_ids, csrc_ids, cdst_ids, formats
        )

        # 确保内存连续（拷贝开销极小，但避免后续 CUDA kernel 非法访问）
        self._src_ids = src_ids.contiguous() if src_ids is not None else None
        self._dst_ids = dst_ids.contiguous() if dst_ids is not None else None
        self._csrc_ids = csrc_ids.contiguous() if csrc_ids is not None else None
        self._cdst_ids = cdst_ids.contiguous() if cdst_ids is not None else None
        self._values = values.contiguous() if values is not None else None
        self._perm_coo2csc = None
        self._perm_csc2csr = None

        if isinstance(formats, str):
            formats = (formats,)
        self._formats = formats

        # 20% 改写：CSC 建立独立为 _build_csc()
        if self._cdst_ids is None:
            self._build_csc()

        # 预建其他请求格式
        for fmt in formats:
            assert fmt in SparseGraph.supported_formats, (
                f"不支持的格式: '{fmt}'。"
                f" 可选: {list(SparseGraph.supported_formats.keys())}。"
            )
            getattr(self, fmt)()  # 触发 lazy 构建

        _dbg(
            "init",
            f"SparseGraph 初始化完成 | size={size} nnz={self._src_ids.numel()} "
            f"formats={formats} is_sorted={is_sorted} "
            f"has_values={values is not None} "
            f"reduce_memory={reduce_memory}",
        )

        self._reduce_memory = reduce_memory
        if reduce_memory:
            self.reduce_memory()

    # ------------------------------------------------------------------
    # 20% 改写：_validate_inputs
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        size: Tuple[int, int],
        src_ids,
        dst_ids,
        csrc_ids,
        cdst_ids,
        formats,
    ) -> None:
        """参数校验（上游散落 6 处 if/raise，本版集中）。"""
        if dst_ids is None and cdst_ids is None:
            raise ValueError(
                "SparseGraph 构造需要 'dst_ids' 或 'cdst_ids' 中的至少一个。"
            )

        if csrc_ids is not None and csrc_ids.numel() != size[0] + 1:
            raise RuntimeError(
                # migrate 1295d2f: f-string 内算术加空格（PEP 8 E225）
                f"csrc_ids 尺寸不符: 期望 ({size[0] + 1},), "
                f"实际 {tuple(csrc_ids.size())}。"
            )

        if cdst_ids is not None and cdst_ids.numel() != size[1] + 1:
            raise RuntimeError(
                # migrate 1295d2f: f-string 内算术加空格（PEP 8 E225）
                f"cdst_ids 尺寸不符: 期望 ({size[1] + 1},), "
                f"实际 {tuple(cdst_ids.size())}。"
            )

        fmt_tuple = (formats,) if isinstance(formats, str) else formats
        if "csc" not in fmt_tuple:
            raise ValueError(
                f"SparseGraph.formats 必须包含 'csc', 实际传入了 {fmt_tuple}。"
            )

        _dbg(
            "validate",
            f"校验通过 | size={size} "
            f"src_ids={None if src_ids is None else tuple(src_ids.shape)} "
            f"dst_ids={None if dst_ids is None else tuple(dst_ids.shape)} "
            f"cdst_ids={None if cdst_ids is None else tuple(cdst_ids.shape)} "
            f"formats={fmt_tuple}",
        )

    # ------------------------------------------------------------------
    # 20% 改写：_build_csc — 把 sort+compress 路径独立出来
    # ------------------------------------------------------------------

    def _build_csc(self) -> None:
        """首次建立 CSC 表示：对 dst 排序，然后压缩为指针数组。"""
        _dbg("build_csc", "开始排序 dst_ids 以建立 CSC…")
        t0 = _time.monotonic()

        if not self._is_sorted:
            self._dst_ids, self._perm_coo2csc = torch.sort(self._dst_ids)
            self._src_ids = self._src_ids[self._perm_coo2csc]
            if self._values is not None:
                self._values = self._values[self._perm_coo2csc]

        self._cdst_ids = compress_ids(self._dst_ids, self._num_dst_nodes)

        _dbg(
            "build_csc",
            f"CSC 建立完毕 | cdst_ids.shape={tuple(self._cdst_ids.shape)} "
            f"耗时={(_time.monotonic()-t0)*1000:.1f}ms",
        )

    # ------------------------------------------------------------------
    # 内存整理
    # ------------------------------------------------------------------

    def reduce_memory(self) -> None:
        """释放当前格式集合不需要的中间张量，降低显存占用。"""
        if self._formats is None:
            return

        needed = set()
        for fmt in self._formats:
            needed.update(SparseGraph.supported_formats[fmt])

        freed = []
        for attr in SparseGraph.all_tensors.difference(needed):
            if self.__dict__.get(attr) is not None:
                self.__dict__[attr] = None
                freed.append(attr)

        _dbg("reduce_memory", f"已释放张量: {freed}")

    # ------------------------------------------------------------------
    # 访问器（CSC 始终可用；COO/CSR 按需 lazy 建立）
    # ------------------------------------------------------------------

    def src_ids(self) -> "torch.Tensor":
        _dbg("src_ids", f"shape={tuple(self._src_ids.shape)}")
        return self._src_ids

    def cdst_ids(self) -> "torch.Tensor":
        _dbg("cdst_ids", f"shape={tuple(self._cdst_ids.shape)}")
        return self._cdst_ids

    def dst_ids(self) -> "torch.Tensor":
        if self._dst_ids is None:
            _dbg("dst_ids", "lazy 解压缩 cdst_ids → dst_ids")
            self._dst_ids = decompress_ids(self._cdst_ids)
        _dbg("dst_ids", f"shape={tuple(self._dst_ids.shape)}")
        return self._dst_ids

    def csrc_ids(self) -> "torch.Tensor":
        if self._csrc_ids is None:
            _dbg("csrc_ids", "lazy 从 src_ids 排序建立 CSR 指针")
            src_ids_sorted, self._perm_csc2csr = torch.sort(self._src_ids)
            self._csrc_ids = compress_ids(src_ids_sorted, self._num_src_nodes)
        _dbg("csrc_ids", f"shape={tuple(self._csrc_ids.shape)}")
        return self._csrc_ids

    def num_src_nodes(self) -> int:
        return self._num_src_nodes

    def num_dst_nodes(self) -> int:
        return self._num_dst_nodes

    def values(self) -> Optional["torch.Tensor"]:
        return self._values

    def formats(self) -> Tuple[str, ...]:
        return self._formats

    # ------------------------------------------------------------------
    # 格式获取（返回三元组 (offsets/row_ptr, col_idx/src_ids, values)）
    # ------------------------------------------------------------------

    def coo(self) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """返回 (src_ids, dst_ids, values) 三元组（COO 格式）。"""
        if "coo" not in self.formats():
            raise RuntimeError(
                "SparseGraph 未预建 COO 格式。"
                " 构造时将 'coo' 加入 formats 参数。"
            )
        return self.src_ids(), self.dst_ids(), self._values

    def csc(self) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """返回 (cdst_ids, src_ids, values) 三元组（CSC 格式）。"""
        if "csc" not in self.formats():
            raise RuntimeError(
                "SparseGraph 未预建 CSC 格式。"
                " 构造时将 'csc' 加入 formats 参数。"
            )
        return self.cdst_ids(), self.src_ids(), self._values

    def csr(self) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """返回 (csrc_ids, dst_ids, values) 三元组（CSR 格式）。"""
        if "csr" not in self.formats():
            raise RuntimeError(
                "SparseGraph 未预建 CSR 格式。"
                " 构造时将 'csr' 加入 formats 参数。"
            )
        csrc = self.csrc_ids()
        dst = self.dst_ids()[self._perm_csc2csr]
        val = self._values
        if val is not None:
            val = val[self._perm_csc2csr]
        return csrc, dst, val

    # ------------------------------------------------------------------
    # 魔术方法
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nnz = self._src_ids.size(0) if self._src_ids is not None else "?"
        return (
            f"{self.__class__.__name__}("
            f"num_src_nodes={self._num_src_nodes}, "
            f"num_dst_nodes={self._num_dst_nodes}, "
            f"num_edges={nnz}, "
            f"formats={self._formats})"
        )
