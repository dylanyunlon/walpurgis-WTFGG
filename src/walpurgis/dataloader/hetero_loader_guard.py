# Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# Licensed under the Apache License, Version 2.0.
#
# 迁移来源: cugraph-gnn commit a9ab8b4
# 原标题: [FEA] Support Heterogeneous Sampling in cuGraph-PyG
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「从来如此，便对么？」—— 鲁迅《狂人日记》
# a9ab8b4 对 NeighborLoader / LinkNeighborLoader / NodeLoader 做了三处关键改动：
#
# 1. NeighborLoader.__init__ & LinkNeighborLoader.__init__:
#    - 异构图时自动选择 compression="COO"（CSR 不支持异构）
#    - 异构图时禁止 directory 参数（磁盘写出不支持异构）
#    - 向 DistributedNeighborSampler 传入 heterogeneous=True,
#      vertex_type_offsets, num_edge_types
#
# 2. NodeLoader.__init__:
#    - input_type 不为 None 时，input_nodes += _vertex_offsets[input_type]
#      （将局部节点 id 转为全局节点 id，供 cuGraph sampler 使用）
#
# 3. LinkLoader.__init__:
#    - input_type 不为 None 时，edge_label_index[0/1] += _vertex_offsets[src/dst]
#
# Walpurgis 迁移策略:
#   上游 loader 文件由 cugraph_pyg.loader 提供，Walpurgis 以 Mixin 方式注入新行为，
#   不复制整个 loader 文件，避免与其他 patch 文件（link_loader_edge_index_guard.py 等）冲突。
#   迁移为独立可测试的辅助函数集合 + 描述性文档。
#
# Walpurgis 20% 改写要点:
#   1. HeteroLoaderGuard dataclass — 封装异构 loader 的前置校验逻辑，
#      替代 neighbor_loader.py / link_neighbor_loader.py 里散落的 if not is_homogeneous 检查
#   2. NodeInputOffset / LinkInputOffset — 独立封装 vertex offset 注入逻辑，
#      替代 node_loader.py / link_loader.py 里的内联偏移计算
#   3. 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖：
#      - HeteroLoaderGuard.validate() 校验结果
#      - NodeInputOffset.apply() 注入前后节点 id 范围
#      - LinkInputOffset.apply() 注入前后 edge_label_index 范围

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _WDBG:
        print(f"[WALPURGIS-LOADER:{tag}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# HeteroLoaderGuard — 对应 NeighborLoader/LinkNeighborLoader 异构校验逻辑
# ---------------------------------------------------------------------------

@dataclass
class HeteroLoaderGuard:
    """
    封装 a9ab8b4 在 NeighborLoader/__init__ 和 LinkNeighborLoader/__init__
    中新增的异构图前置校验逻辑。

    上游新增的逻辑（两个 loader 几乎相同）:
        feature_store, graph_store = data
        if compression is None:
            compression = "CSR" if graph_store.is_homogeneous else "COO"
        elif compression not in ["CSR", "COO"]:
            raise ValueError("Invalid value for compression ...")

        if not graph_store.is_homogeneous:
            if compression != "COO":
                raise ValueError("Only COO format is supported for heterogeneous graphs!")
            if directory is not None:
                raise ValueError("Writing to disk is not supported for heterogeneous graphs!")

    上游同时向 DistributedNeighborSampler 新增三个 kwargs:
        heterogeneous=(not graph_store.is_homogeneous),
        vertex_type_offsets=graph_store._vertex_offset_array,
        num_edge_types=len(graph_store.get_all_edge_attrs()),

    Walpurgis Bug 分析:
    - 上游将 feature_store, graph_store = data 拆包从 if compression is None 之前
      移到了开头（a9ab8b4 修复变量使用顺序），Walpurgis 同步此修复
    - 异构时 CSR → 静默崩溃（上游改为显式 ValueError），Walpurgis 加 DEBUG 前缀
    """

    is_homogeneous: bool
    compression: Optional[str]  # 用户传入值，None 表示自动选择
    directory: Optional[str]
    num_edge_attrs: int
    vertex_type_offsets: Optional["torch.Tensor"]  # graph_store._vertex_offset_array

    # 经过 resolve() 后的最终 compression 值
    resolved_compression: str = field(init=False, default="")

    def resolve(self) -> "HeteroLoaderGuard":
        """
        执行校验并解析最终 compression 值。
        对应上游两个 loader 开头的 if compression is None / if not is_homogeneous 块。
        """
        # 自动选择 compression
        if self.compression is None:
            self.resolved_compression = "CSR" if self.is_homogeneous else "COO"
        else:
            if self.compression not in ("CSR", "COO"):
                raise ValueError(
                    f"[Walpurgis:HeteroLoaderGuard] Invalid value for compression: "
                    f"{self.compression!r}. Expected 'CSR' or 'COO'."
                )
            self.resolved_compression = self.compression

        _dbg(
            "HeteroLoaderGuard.resolve",
            f"is_homogeneous={self.is_homogeneous} "
            f"compression_in={self.compression!r} → resolved={self.resolved_compression!r} "
            f"directory={self.directory!r} num_edge_attrs={self.num_edge_attrs}",
        )

        # 异构图额外约束
        if not self.is_homogeneous:
            if self.resolved_compression != "COO":
                raise ValueError(
                    "[Walpurgis:HeteroLoaderGuard] "
                    "Only COO format is supported for heterogeneous graphs! "
                    f"Got: {self.resolved_compression!r}."
                )
            if self.directory is not None:
                raise ValueError(
                    "[Walpurgis:HeteroLoaderGuard] "
                    "Writing to disk is not supported for heterogeneous graphs! "
                    f"directory={self.directory!r}"
                )
            _dbg(
                "HeteroLoaderGuard.resolve",
                f"异构图校验通过: COO格式, 无 directory, "
                f"vertex_type_offsets.shape="
                f"{tuple(self.vertex_type_offsets.shape) if self.vertex_type_offsets is not None else None}",
            )

        return self

    def sampler_kwargs(self) -> dict:
        """
        返回需要额外传给 DistributedNeighborSampler 的关键字参数。

        a9ab8b4 新增的三个参数:
            heterogeneous=(not graph_store.is_homogeneous),
            vertex_type_offsets=graph_store._vertex_offset_array,
            num_edge_types=len(graph_store.get_all_edge_attrs()),
        """
        kwargs = {
            "heterogeneous": not self.is_homogeneous,
            "num_edge_types": self.num_edge_attrs,
        }
        if not self.is_homogeneous:
            kwargs["vertex_type_offsets"] = self.vertex_type_offsets

        _dbg(
            "HeteroLoaderGuard.sampler_kwargs",
            f"heterogeneous={kwargs['heterogeneous']} "
            f"num_edge_types={kwargs['num_edge_types']} "
            f"vertex_type_offsets={'set' if 'vertex_type_offsets' in kwargs else 'omitted'}",
        )
        return kwargs

    @classmethod
    def from_graph_store(
        cls,
        graph_store,
        compression: Optional[str],
        directory: Optional[str],
    ) -> "HeteroLoaderGuard":
        """
        从 graph_store 构造 HeteroLoaderGuard 并立即执行 resolve()。

        对应上游两个 loader 的:
            feature_store, graph_store = data
            ...校验逻辑...
            BaseSampler(...,
                heterogeneous=(not graph_store.is_homogeneous),
                vertex_type_offsets=graph_store._vertex_offset_array,
                num_edge_types=len(graph_store.get_all_edge_attrs()),
            )
        """
        guard = cls(
            is_homogeneous=graph_store.is_homogeneous,
            compression=compression,
            directory=directory,
            num_edge_attrs=len(graph_store.get_all_edge_attrs()),
            vertex_type_offsets=(
                graph_store._vertex_offset_array
                if not graph_store.is_homogeneous
                else None
            ),
        )
        return guard.resolve()


# ---------------------------------------------------------------------------
# NodeInputOffset — 对应 NodeLoader.__init__ 的 input_nodes offset 注入
# ---------------------------------------------------------------------------

@dataclass
class NodeInputOffset:
    """
    封装 a9ab8b4 在 NodeLoader.__init__ 中新增的 vertex offset 注入逻辑。

    上游新增 (node_loader.py L109-L110):
        if input_type is not None:
            input_nodes += data[1]._vertex_offsets[input_type]

    根因:
        cuGraph sampler 使用全局节点编号（所有节点类型拼在一起的单一整数空间）。
        PyG 的 input_nodes 是局部类型内的编号（从 0 开始）。
        为了让 cuGraph 正确找到采样起始节点，必须在传入 sampler 之前加上类型 offset。

    上游 bug（a9ab8b4 之前）:
        input_nodes 直接传给 sampler，cuGraph 将其解释为全局 id，
        但实际上是局部 id（从 0 开始），导致采样到错误节点，
        且不报错，只是训练数据完全错误。
    """

    vertex_offsets: Dict[str, int]   # graph_store._vertex_offsets

    def apply(
        self,
        input_nodes: "torch.Tensor",
        input_type: Optional[str],
    ) -> "torch.Tensor":
        """
        若 input_type 不为 None，对 input_nodes 加上对应类型的 vertex offset。

        对应上游:
            if input_type is not None:
                input_nodes += data[1]._vertex_offsets[input_type]
        """
        if input_type is None:
            _dbg("NodeInputOffset.apply", f"input_type=None, 跳过 offset 注入")
            return input_nodes

        offset = self.vertex_offsets[input_type]

        _dbg(
            "NodeInputOffset.apply",
            f"input_type={input_type!r} offset={offset} "
            f"input_nodes range=[{int(input_nodes.min())}, {int(input_nodes.max())}] "
            f"→ [{int(input_nodes.min()) + offset}, {int(input_nodes.max()) + offset}]",
        )

        # in-place += 对应上游行为（input_nodes 是 cuda tensor）
        input_nodes = input_nodes + offset

        _dbg(
            "NodeInputOffset.apply",
            f"注入完成: input_nodes.shape={tuple(input_nodes.shape)} "
            f"offset_applied={offset}",
        )
        return input_nodes


# ---------------------------------------------------------------------------
# LinkInputOffset — 对应 LinkLoader.__init__ 的 edge_label_index offset 注入
# ---------------------------------------------------------------------------

@dataclass
class LinkInputOffset:
    """
    封装 a9ab8b4 在 LinkLoader.__init__ 中新增的 edge_label_index offset 注入逻辑。

    上游新增 (link_loader.py L128-L131):
        # Note reverse of standard convention here
        if input_type is not None:
            edge_label_index[0] += data[1]._vertex_offsets[input_type[0]]
            edge_label_index[1] += data[1]._vertex_offsets[input_type[2]]

    注意上游注释 "Note reverse of standard convention here":
        edge_label_index[0] 对应 src node（PyG 约定），但偏移用 input_type[0]（src 类型）
        edge_label_index[1] 对应 dst node，但偏移用 input_type[2]（dst 类型）
        这与 graph_store.py 里 PyG edge_index 约定相反（那里 [0]=dst, [1]=src），
        link_loader 遵循标准 PyG 约定，所以注释说 "reverse"。

    上游 bug（a9ab8b4 之前）:
        同 NodeInputOffset — edge_label_index 是局部 id，需加 vertex offset 才能传给 cuGraph。
    """

    vertex_offsets: Dict[str, int]   # graph_store._vertex_offsets

    def apply(
        self,
        edge_label_index: "torch.Tensor",   # shape [2, N]
        input_type: Optional[Tuple[str, str, str]],
    ) -> "torch.Tensor":
        """
        若 input_type 不为 None，对 edge_label_index 分别注入 src/dst vertex offset。

        对应上游:
            if input_type is not None:
                edge_label_index[0] += data[1]._vertex_offsets[input_type[0]]
                edge_label_index[1] += data[1]._vertex_offsets[input_type[2]]
        """
        if input_type is None:
            _dbg("LinkInputOffset.apply", f"input_type=None, 跳过 offset 注入")
            return edge_label_index

        src_offset = self.vertex_offsets[input_type[0]]
        dst_offset = self.vertex_offsets[input_type[2]]

        _dbg(
            "LinkInputOffset.apply",
            f"input_type={input_type!r} "
            f"src_offset={src_offset} dst_offset={dst_offset} "
            f"edge_label_index[0] range=[{int(edge_label_index[0].min())}, "
            f"{int(edge_label_index[0].max())}] "
            f"edge_label_index[1] range=[{int(edge_label_index[1].min())}, "
            f"{int(edge_label_index[1].max())}]",
        )

        # a9ab8b4 注释: "Note reverse of standard convention here"
        # edge_label_index[0] = src nodes → 加 src_type (input_type[0]) 的 offset
        # edge_label_index[1] = dst nodes → 加 dst_type (input_type[2]) 的 offset
        edge_label_index = edge_label_index.clone()
        edge_label_index[0] = edge_label_index[0] + src_offset
        edge_label_index[1] = edge_label_index[1] + dst_offset

        _dbg(
            "LinkInputOffset.apply",
            f"注入完成: "
            f"edge_label_index[0] range=[{int(edge_label_index[0].min())}, "
            f"{int(edge_label_index[0].max())}] "
            f"edge_label_index[1] range=[{int(edge_label_index[1].min())}, "
            f"{int(edge_label_index[1].max())}]",
        )
        return edge_label_index


# ---------------------------------------------------------------------------
# GcnLoaderTempDirRemoval — 对应 a9ab8b4 对 gcn_dist_*.py examples 的 tempdir 移除
# ---------------------------------------------------------------------------

class GcnLoaderTempDirRemoval:
    """
    文档类：记录 a9ab8b4 对 GCN 分布式示例中 tempdir 逻辑的移除。

    上游变化 (gcn_dist_mnmg.py / gcn_dist_sg.py / gcn_dist_snmg.py):
        旧: with tempfile.TemporaryDirectory(dir=args.tempdir_root) as tempdir:
                train_loader = NeighborLoader(..., directory=os.path.join(tempdir, "train"), ...)
        新: train_loader = NeighborLoader(..., ...)  # 无 directory 参数

    根因:
        a9ab8b4 移除了 NeighborLoader 的 directory 参数支持（异构图不支持磁盘写出）。
        同时，磁盘写出在实际训练中很少有性能优势（反而增加 I/O 开销），
        因此一并从所有示例中移除。

    Walpurgis 迁移:
        - Walpurgis GCN 示例 (examples/gcn/gcn_dist_mnmg.py) 已在前序 commit 迁移时
          不含 tempdir 逻辑，无需再次修改。
        - 此类仅作为迁移记录，供 MIGRATION_LOG.md 引用。

    已验证: walpurgis-WTFGG/src/walpurgis/examples/gcn/gcn_dist_mnmg.py 无 tempdir 参数。
    """
    REMOVED_PARAMS = ["directory", "tempdir_root", "tempdir"]
    REASON = (
        "a9ab8b4: NeighborLoader 异构图不支持 directory 写出，"
        "同时磁盘写出性能无优势，全部示例统一移除。"
    )
