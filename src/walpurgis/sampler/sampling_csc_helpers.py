# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate db74d87: Merge pull request #2 from alexbarghi-nv/copy-from-cugraph
# 上游源: python/cugraph-dgl/cugraph_dgl/dataloading/utils/sampling_helpers.py
#         函数: _process_sampled_df_csc, _create_homogeneous_sparse_graphs_from_csc,
#               create_homogeneous_sampled_graphs_from_dataframe_csc
# Walpurgis 迁移: BulkSampler CSC 压缩格式采样结果后处理
#
# 鲁迅《且介亭杂文·拿来主义》：
# 「总之，我们要拿来。我们要或使用，或存放，或毁灭。」
# 上游 _process_sampled_df_csc 把 numpy 转换、局部偏移修正、renumber_map 提取
# 全部塞进一个两百行函数——没有任何中间状态可观察，也没有步骤名。
# 本版按职责拆成三个内聚函数，每步加断点，让 CSC 流水线的每个阶段有迹可循。
#
# 20% 改写要点（保持上游公开 API `create_homogeneous_sampled_graphs_from_dataframe_csc` 完全兼容）：
#   1. _extract_csc_tensors() — 专门负责从 DataFrame 提取并局部化偏移，
#      替代原版前半段隐式的 magic index 算术
#   2. _build_per_batch_hop_dict() — 把"按 batch+hop 切分张量"独立出来,
#      原版内联的双重 for 循环在 DEBUG 时无法逐步观察中间值
#   3. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - _extract_csc_tensors：n_batches/n_hops/偏移量修正前后
#      - _build_per_batch_hop_dict：每个 batch 的 hop 切片范围
#      - _create_homogeneous_sparse_graphs_from_csc：每个 batch 的 SparseGraph 尺寸
#      - create_homogeneous_sampled_graphs_from_dataframe_csc：入口 DataFrame 行数

import os as _os
import sys as _sys
import time as _time
from typing import Dict, List, Tuple

import cudf

from walpurgis.utils.imports import import_optional
from walpurgis.tensor.sparse_graph import SparseGraph

torch = import_optional("torch")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-SAMPLE-CSC:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


def _cast_to_tensor(ser: "cudf.Series") -> "torch.Tensor":
    """cudf.Series → GPU torch.Tensor（空 Series 安全转换）。

    上游版本（cast_to_tensor）在 len==0 时调 .values.get() 走 CPU 路径再 .to("cuda")，
    非空时直接 torch.as_tensor(ser.values, device="cuda")。本版保留相同语义。
    """
    if len(ser) == 0:
        t = torch.from_numpy(ser.values.get())
        return t.to("cuda")
    return torch.as_tensor(ser.values, device="cuda")


# ---------------------------------------------------------------------------
# 20% 改写：_extract_csc_tensors
# ---------------------------------------------------------------------------

def _extract_csc_tensors(
    df: "cudf.DataFrame",
) -> Tuple[
    "torch.Tensor",  # major_offsets (int32, cuda)
    "torch.Tensor",  # minors (int32, cuda)
    "torch.Tensor",  # label_hop_offsets (int64, cuda)
    "torch.Tensor",  # renumber_map (cuda)
    "torch.Tensor",  # renumber_map_offsets (int64, cuda)
    int,             # n_batches
    int,             # n_hops
]:
    """从 BulkSampler CSC 压缩格式 DataFrame 提取核心张量，并将全局偏移局部化。

    上游原版把这段逻辑散在 _process_sampled_df_csc 中，没有命名——
    本版将其独立，方便单测和 DEBUG 逐步观察。

    DataFrame 约定（BulkSampler compression="CSR" 输出，因采样以源为 major）：
        columns: major_offsets, minors, label_hop_offsets,
                 renumber_map_offsets, map
    """
    _dbg("extract", f"DataFrame shape={df.shape}")

    # 提取并去 NaN
    major_offsets = _cast_to_tensor(df.major_offsets.dropna())
    label_hop_offsets = _cast_to_tensor(df.label_hop_offsets.dropna())
    renumber_map_offsets = _cast_to_tensor(df.renumber_map_offsets.dropna())
    renumber_map = _cast_to_tensor(df["map"].dropna())
    minors = _cast_to_tensor(df.minors.dropna())

    n_batches = len(renumber_map_offsets) - 1
    n_hops = int((len(label_hop_offsets) - 1) / n_batches)

    _dbg(
        "extract",
        f"n_batches={n_batches} n_hops={n_hops} "
        f"major_offsets.numel={major_offsets.numel()} "
        f"minors.numel={minors.numel()} "
        f"renumber_map.numel={renumber_map.numel()}",
    )

    # 将全局偏移局部化（原版用 clone() 避免 in-place 写入 autograd 图）
    major_offsets = major_offsets - major_offsets[0].clone()
    label_hop_offsets = label_hop_offsets - label_hop_offsets[0].clone()
    renumber_map_offsets = renumber_map_offsets - renumber_map_offsets[0].clone()

    _dbg(
        "extract",
        f"偏移局部化完成 | "
        f"major_offsets[0]={int(major_offsets[0])} "
        f"label_hop_offsets[0]={int(label_hop_offsets[0])}",
    )

    # 转 int32（pylibcugraphops binding 要求 minors/major_offsets 同类型）
    minors = minors.int()
    major_offsets = major_offsets.int()

    return (
        major_offsets,
        minors,
        label_hop_offsets,
        renumber_map,
        renumber_map_offsets,
        n_batches,
        n_hops,
    )


# ---------------------------------------------------------------------------
# 20% 改写：_build_per_batch_hop_dict
# ---------------------------------------------------------------------------

def _build_per_batch_hop_dict(
    major_offsets: "torch.Tensor",
    minors: "torch.Tensor",
    label_hop_offsets: "torch.Tensor",
    renumber_map: "torch.Tensor",
    renumber_map_offsets: "torch.Tensor",
    n_batches: int,
    n_hops: int,
    reverse_hop_id: bool,
) -> Tuple[
    Dict[int, Dict[int, Dict[str, "torch.Tensor"]]],  # tensors_dict
    List["torch.Tensor"],                              # renumber_map_list
    List[List[int]],                                   # mfg_sizes
]:
    """按 batch/hop 切分张量，构建 SparseGraph 所需的输入结构。

    上游原版用 numpy + 双重 for 循环在 Python 层完成切片，
    本版保留相同算法，但加调试输出帮助确认每个 batch 的切片范围。
    """
    # 计算每个 MFG 层的节点数（用于 SparseGraph size 参数）
    mfg_sizes = (label_hop_offsets[1:] - label_hop_offsets[:-1]).reshape(
        (n_batches, n_hops)
    )
    n_nodes = renumber_map_offsets[1:] - renumber_map_offsets[:-1]
    mfg_sizes = torch.hstack((mfg_sizes, n_nodes.reshape(n_batches, -1)))
    if reverse_hop_id:
        mfg_sizes = mfg_sizes.flip(1)

    # 转 CPU numpy 用于 Python 索引（批量转换避免逐元素传输开销）
    major_offsets_cpu = major_offsets.to("cpu").numpy()
    label_hop_offsets_cpu = label_hop_offsets.to("cpu").numpy()

    tensors_dict: Dict[int, Dict] = {}
    renumber_map_list: List["torch.Tensor"] = []

    for batch_id in range(n_batches):
        batch_dict: Dict[int, Dict[str, "torch.Tensor"]] = {}

        for hop_id in range(n_hops):
            idx = batch_id * n_hops + hop_id  # 在 label_hop_offsets 中的绝对位置
            mo_start = int(label_hop_offsets_cpu[idx])
            mo_end = int(label_hop_offsets_cpu[idx + 1])
            mi_start = int(major_offsets_cpu[mo_start])
            mi_end = int(major_offsets_cpu[mo_end])

            hop_dict: Dict[str, "torch.Tensor"] = {
                "minors": minors[mi_start:mi_end],
                "major_offsets": (
                    major_offsets[mo_start : mo_end + 1]
                    - major_offsets[mo_start]
                ),
            }

            _dbg(
                "per_batch_hop",
                f"batch={batch_id} hop={hop_id} | "
                f"mo_slice=[{mo_start},{mo_end}] "
                f"mi_slice=[{mi_start},{mi_end}] "
                f"minors.numel={hop_dict['minors'].numel()} "
                f"major_offsets.numel={hop_dict['major_offsets'].numel()}",
            )

            target_hop = (n_hops - 1 - hop_id) if reverse_hop_id else hop_id
            batch_dict[target_hop] = hop_dict

        tensors_dict[batch_id] = batch_dict
        renumber_map_list.append(
            renumber_map[
                int(renumber_map_offsets[batch_id]) : int(renumber_map_offsets[batch_id + 1])
            ]
        )

    return tensors_dict, renumber_map_list, mfg_sizes.tolist()


# ---------------------------------------------------------------------------
# 上游公开接口（保持原名以便上游测试可直接调用）
# ---------------------------------------------------------------------------

def _process_sampled_df_csc(
    df: "cudf.DataFrame",
    reverse_hop_id: bool = True,
) -> Tuple[
    Dict[int, Dict[int, Dict[str, "torch.Tensor"]]],
    List["torch.Tensor"],
    List[List[int]],
]:
    """将 BulkSampler CSC 格式 DataFrame 转换为张量字典 + renumber_map + mfg_sizes。

    此函数是上游 _process_sampled_df_csc 的 Walpurgis 版本。
    内部用 _extract_csc_tensors + _build_per_batch_hop_dict 替代原版的单体实现。

    Parameters
    ----------
    df: cudf.DataFrame
        BulkSampler 输出（compression="CSR"，采样以源为 major）。
    reverse_hop_id: bool
        是否翻转 hop 顺序（默认 True，与 DGL 消息传递方向对齐）。

    Returns
    -------
    tensors_dict: dict
        ``tensors_dict[batch_id][hop_id]`` 包含 ``"minors"`` 和 ``"major_offsets"``。
    renumber_map_list: list of Tensor
        每个 batch 的重编号映射。
    mfg_sizes: list
        每个 batch 各层节点数，shape (n_batches, n_hops+1)。
    """
    _dbg("process_df_csc", f"入口 | df.shape={df.shape}")

    (
        major_offsets,
        minors,
        label_hop_offsets,
        renumber_map,
        renumber_map_offsets,
        n_batches,
        n_hops,
    ) = _extract_csc_tensors(df)

    return _build_per_batch_hop_dict(
        major_offsets,
        minors,
        label_hop_offsets,
        renumber_map,
        renumber_map_offsets,
        n_batches,
        n_hops,
        reverse_hop_id=reverse_hop_id,
    )


def _create_homogeneous_sparse_graphs_from_csc(
    tensors_dict: Dict[int, Dict[int, Dict[str, "torch.Tensor"]]],
    renumber_map_list: List["torch.Tensor"],
    mfg_sizes: List[List[int]],
) -> List[List]:
    """从张量字典创建 mini-batch MFG 列表（每个 MFG 用 SparseGraph 表示）。

    Parameters
    ----------
    tensors_dict, renumber_map_list, mfg_sizes:
        ``_process_sampled_df_csc`` 的三个返回值。

    Returns
    -------
    output: list
        每个元素是 [input_nodes, output_nodes, mfgs]，
        其中 mfgs 是当前 batch 各 hop 的 SparseGraph 列表。
    """
    n_batches = len(mfg_sizes)
    n_hops = len(mfg_sizes[0]) - 1

    _dbg(
        "build_sparse_graphs",
        f"n_batches={n_batches} n_hops={n_hops}",
    )

    output = []
    for b_id in range(n_batches):
        mfgs = []
        for h_id in range(n_hops):
            num_src = int(mfg_sizes[b_id][h_id])
            num_dst = int(mfg_sizes[b_id][h_id + 1])

            _dbg(
                "build_sparse_graphs",
                f"batch={b_id} hop={h_id} | "
                f"size=({num_src}, {num_dst}) "
                f"minors.numel={tensors_dict[b_id][h_id]['minors'].numel()}",
            )

            sg = SparseGraph(
                size=(num_src, num_dst),
                src_ids=tensors_dict[b_id][h_id]["minors"],
                cdst_ids=tensors_dict[b_id][h_id]["major_offsets"],
                formats=["csc"],
                reduce_memory=True,
            )
            mfgs.append(sg)

        # input_nodes = 整个 renumber_map（最大前缀），output_nodes = 前 n_dst 个节点
        input_nodes = renumber_map_list[b_id]
        output_nodes = renumber_map_list[b_id][: int(mfg_sizes[b_id][-1])]

        output.append([input_nodes, output_nodes, mfgs])

    return output


def create_homogeneous_sampled_graphs_from_dataframe_csc(
    sampled_df: "cudf.DataFrame",
) -> List[List]:
    """公开接口：从 BulkSampler CSC DataFrame 创建同构图 mini-batch MFG 列表。

    上游 cugraph-dgl 同名函数的 Walpurgis 版本。
    内部调用 _process_sampled_df_csc + _create_homogeneous_sparse_graphs_from_csc。

    Parameters
    ----------
    sampled_df: cudf.DataFrame
        BulkSampler 输出（compression="CSR"）。

    Returns
    -------
    output: list
        每个元素 [input_nodes, output_nodes, mfgs]，mfgs 中每个元素是 SparseGraph。
    """
    _dbg(
        "public_api",
        f"入口 | sampled_df.shape={sampled_df.shape}",
    )
    return _create_homogeneous_sparse_graphs_from_csc(
        *(_process_sampled_df_csc(sampled_df))
    )


# ===========================================================================
# f4ca484 迁移新增：基于张量的 CSC 处理函数
# ===========================================================================
# 迁移来源: cugraph-gnn commit f4ca484
# 原位置: cugraph_dgl/dataloading/utils/sampling_helpers.py
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。」
# —— 鲁迅《纪念刘和珍君》
#
# f4ca484 将 _process_sampled_df_csc 重构为两层：
# - _process_sampled_tensors_csc：直接接受 Dict[torch.Tensor]（新路径）
# - _process_sampled_df_csc：将 DataFrame 转为 tensors 后调用上面的函数（兼容旧路径）
# 同时新增 _create_homogeneous_blocks_from_csc（dgl.Block 格式输出）
# 和 create_homogeneous_sampled_graphs_from_tensors_csc（tensor 路径公开入口）
#
# Walpurgis 改写：延续已有 _dbg 模式，添加同等粒度的断点。


def _process_sampled_tensors_csc(
    tensors: Dict["torch.Tensor"],
    reverse_hop_id: bool = True,
):
    """
    f4ca484 新增：从 Dict[torch.Tensor] 转换 CSC 采样输出（不依赖 DataFrame）。

    原版的 _process_sampled_df_csc 现在通过此函数实现，
    此函数也被 HomogeneousSampleReader（非 dask 路径）直接调用。

    Parameters
    ----------
    tensors : Dict[torch.Tensor]
        BulkSampler / UniformNeighborSampler 压缩 CSC 格式输出的张量字典。
        必须包含: major_offsets / minors / label_hop_offsets /
                 map / renumber_map_offsets
    reverse_hop_id : bool (default=True)
        是否翻转 hop id 顺序。

    Returns
    -------
    与 _process_sampled_df_csc 相同的三元组:
        (tensors_dict, renumber_map_list, mfg_sizes)
    """
    _dbg(
        "_process_sampled_tensors_csc",
        f"keys={list(tensors.keys())}",
    )

    major_offsets = tensors["major_offsets"]
    minors = tensors["minors"]
    label_hop_offsets = tensors["label_hop_offsets"]
    renumber_map = tensors["map"]
    renumber_map_offsets = tensors["renumber_map_offsets"]

    # 下面复用 _build_per_batch_hop_dict 和后续流程，
    # 因此构造与 _process_sampled_df_csc 相同的中间格式
    n_batches = len(renumber_map_offsets) - 1
    n_hops = int((len(label_hop_offsets) - 1) / n_batches)

    _dbg(
        "_process_sampled_tensors_csc",
        f"n_batches={n_batches} n_hops={n_hops} "
        f"major_offsets.numel={major_offsets.numel()} "
        f"minors.numel={minors.numel()}",
    )

    tensors_dict = {}
    renumber_map_list = []
    mfg_sizes = []

    for b_id in range(n_batches):
        tensors_dict[b_id] = {}

        # renumber_map 切片
        map_start = renumber_map_offsets[b_id]
        map_end = renumber_map_offsets[b_id + 1]
        renumber_map_list.append(renumber_map[map_start:map_end])

        batch_sizes = [None] * (n_hops + 1)

        for h_id in range(n_hops):
            if reverse_hop_id:
                hop = n_hops - 1 - h_id
            else:
                hop = h_id

            lho_start = b_id * n_hops + h_id
            lho_end = b_id * n_hops + h_id + 2

            mo_start = label_hop_offsets[lho_start]
            mo_end = label_hop_offsets[lho_end - 1]

            batch_major_offsets = major_offsets[mo_start : mo_end + 1].clone()
            batch_minors = minors[
                batch_major_offsets[0] : batch_major_offsets[-1]
            ]
            batch_major_offsets -= batch_major_offsets[0].clone()

            _dbg(
                "_process_sampled_tensors_csc",
                f"batch={b_id} hop={hop} | "
                f"mo_range=[{int(mo_start)}, {int(mo_end)}] "
                f"minors.numel={batch_minors.numel()}",
            )

            tensors_dict[b_id][hop] = {
                "major_offsets": batch_major_offsets,
                "minors": batch_minors,
            }

            if batch_sizes[hop] is None:
                batch_sizes[hop] = int(batch_major_offsets.numel()) - 1
            if batch_sizes[hop + 1] is None:
                batch_sizes[hop + 1] = (
                    int(batch_minors.max()) + 1 if batch_minors.numel() > 0 else 0
                )

        mfg_sizes.append(batch_sizes)

    return tensors_dict, renumber_map_list, mfg_sizes


def _create_homogeneous_blocks_from_csc(
    tensors_dict: Dict[int, Dict[int, Dict[str, "torch.Tensor"]]],
    renumber_map_list: List["torch.Tensor"],
    mfg_sizes: List,
):
    """
    f4ca484 新增：从 CSC 张量字典创建 dgl.Block 格式的 mini-batch。

    参数与 _create_homogeneous_sparse_graphs_from_csc 相同，
    输出 blocks 为 dgl.Block 而非 SparseGraph。

    Returns
    -------
    output : list
        每个元素 [input_nodes, output_nodes, blocks_list]。
    """
    try:
        import dgl as _dgl
    except ImportError:
        raise ImportError(
            "[Walpurgis] _create_homogeneous_blocks_from_csc 需要 dgl，"
            "但当前环境中未安装。"
        )

    n_batches = len(mfg_sizes)
    n_hops = len(mfg_sizes[0]) - 1 if n_batches > 0 else 0
    output = []

    for b_id in range(n_batches):
        mfgs_sparse = [
            SparseGraph(
                size=(mfg_sizes[b_id][h_id], mfg_sizes[b_id][h_id + 1]),
                src_ids=tensors_dict[b_id][h_id]["minors"],
                cdst_ids=tensors_dict[b_id][h_id]["major_offsets"],
                formats=["csc", "coo"],
                reduce_memory=True,
            )
            for h_id in range(n_hops)
        ]

        _dbg(
            "_create_homogeneous_blocks_from_csc",
            f"batch={b_id} n_hops={n_hops} sizes={mfg_sizes[b_id]}",
        )

        blocks = []
        for mfg in reversed(mfgs_sparse):
            # 转换为 dgl.Block
            num_src = mfg._num_src_nodes
            num_dst = mfg._num_dst_nodes
            src_ids = mfg.src_ids()
            dst_ids = mfg.dst_ids()

            block = _dgl.create_block(
                (src_ids.cpu(), dst_ids.cpu()),
                num_src_nodes=num_src,
                num_dst_nodes=num_dst,
            )
            blocks.append(block)

        del mfgs_sparse
        blocks.reverse()

        output.append([
            renumber_map_list[b_id],
            renumber_map_list[b_id][: mfg_sizes[b_id][-1]],
            blocks,
        ])

    return output


def create_homogeneous_sampled_graphs_from_tensors_csc(
    tensors: Dict["torch.Tensor"],
    output_format: str = "cugraph_dgl.nn.SparseGraph",
) -> List[List]:
    """
    f4ca484 新增公开接口：从张量字典创建同构图 mini-batch。

    与 create_homogeneous_sampled_graphs_from_dataframe_csc 类似，
    但直接接受 Dict[torch.Tensor] 而非 cudf.DataFrame。
    供 HomogeneousSampleReader（非 dask 路径）调用。

    Parameters
    ----------
    tensors : Dict[torch.Tensor]
        UniformNeighborSampler 输出的张量字典。
    output_format : str (default='cugraph_dgl.nn.SparseGraph')
        输出格式：'cugraph_dgl.nn.SparseGraph' 或 'dgl.Block'。

    Returns
    -------
    list
        每个元素 [input_nodes, output_nodes, mfgs_or_blocks]。
    """
    _dbg(
        "create_homogeneous_sampled_graphs_from_tensors_csc",
        f"output_format={output_format!r} keys={list(tensors.keys())}",
    )

    if output_format == "cugraph_dgl.nn.SparseGraph":
        return _create_homogeneous_sparse_graphs_from_csc(
            *(_process_sampled_tensors_csc(tensors))
        )
    elif output_format == "dgl.Block":
        return _create_homogeneous_blocks_from_csc(
            *(_process_sampled_tensors_csc(tensors))
        )
    else:
        raise ValueError(
            f"[Walpurgis] 无效 output_format={output_format!r}，"
            f"期望 'cugraph_dgl.nn.SparseGraph' 或 'dgl.Block'。"
        )
