"""
sampler_utils.py — dd543dc 迁移: 负采样修复 + 采样结果处理工具

migrate dd543dc: Heterogeneous Link Prediction Example for cuGraph-PyG

上游变化核心 (dd543dc, cugraph-gnn / python/cugraph-pyg/cugraph_pyg/sampler/sampler_utils.py):

1. neg_sample() 删除错误的分布式 all_reduce:
   旧代码:
       if graph_store.is_multi_gpu:
           num_neg_global = torch.tensor([num_neg], device="cuda")
           torch.distributed.all_reduce(num_neg_global, op=torch.distributed.ReduceOp.SUM)
           num_neg = int(num_neg_global)
       else:
           num_neg_global = num_neg
       result_dict = pylibcugraph.negative_sampling(..., num_neg_global, ...)
   新代码:
       result_dict = pylibcugraph.negative_sampling(..., num_neg, ...)
   Bug 根因:
       多 GPU 时 all_reduce SUM 把每个 rank 各自需要的 num_neg 相加，
       导致每个 rank 都请求 world_size 倍的负样本。
       example: 4 GPU, 每 rank num_neg=100 → all_reduce 后 num_neg=400，
       pylibcugraph 为每 rank 生成 400 个负样本 → 全局 1600 个（应为 400 个）。
       负样本数量膨胀导致正负比严重失衡，链路预测 AUC 下降但不报错（沉默 bug）。
       修复: 每 rank 独立生成 num_neg 个负样本，无需全局对齐。

2. filter_cugraph_pyg_store():
   构建采样后的 torch_geometric.data.Data 对象，
   根据 feature_store.get_all_tensor_attrs() 批量拉取特征。
   边特征用 edge index，节点特征用 node index。

3. _sampler_output_from_sampling_results_*:
   DaskGraphStore 路径（旧 API）的采样结果后处理函数，
   被 walpurgis.sampler.sampler.py 通过延迟 import 调用。

Walpurgis 改写 20%（鲁迅拿法）:
- NegSampler 数据类封装 neg_sample 的参数 + 状态
  上游函数签名 8 个参数散装，NegSampler.__call__ 给参数命名，
  并在 WALPURGIS_DEBUG=1 时打印 num_neg / weighted / 结果 src/dst shape
- SamplerResultValidator 封装 pylibcugraph 结果校验 + fallback 填充
  上游 if src_neg.numel() < num_neg 的 randint fallback 内联在主路径中，
  SamplerResultValidator.validate_and_pad() 给这段逻辑命名，加断点
- HopIndexer dataclass 封装 torch.searchsorted hops 构建
  上游在三个函数中重复同样的 searchsorted 模式，
  HopIndexer 提取为可复用对象，携带 hop_starts / num_hops
- 全链路 WALPURGIS_DEBUG=1 断点 print

作者: dylanyunlon <dogechat@163.com>
"""

import os
import sys
from typing import Sequence, Dict, Tuple, Optional, Union
from math import ceil
from dataclasses import dataclass, field

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 延迟 import（与上游保持一致，允许在无 GPU 环境导入模块）
# ─────────────────────────────────────────────────────────────────────────────

try:
    from cugraph.utilities.utils import import_optional
    import cudf
    import cupy
    import pylibcugraph

    dask_cudf = import_optional("dask_cudf")
    torch_geometric = import_optional("torch_geometric")
    torch = import_optional("torch")
    HeteroSamplerOutput = torch_geometric.sampler.base.HeteroSamplerOutput

    from cugraph_pyg.data import GraphStore, DaskGraphStore
except ImportError:
    # 无 GPU / 单元测试环境
    GraphStore = object
    DaskGraphStore = object
    HeteroSamplerOutput = None
    torch_geometric = None
    torch = None
    cudf = None
    cupy = None
    pylibcugraph = None
    _dbg("import", "GPU dependencies not available — sampler_utils in stub mode")


# ─────────────────────────────────────────────────────────────────────────────
# verify_metadata — migrate 2ba9979: 异构图 metadata 校验
# ─────────────────────────────────────────────────────────────────────────────

def verify_metadata(
    metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]
) -> None:
    """
    校验 metadata 字典的类型约束。

    migrate 2ba9979 核心逻辑:
      上游将 metadata 作为任意 dict 传入 DistributedNeighborSampler，
      但 pylibcugraph 只接受 str 或 (str, str, str) 的值类型。
      若不提前校验，运行时会在 C 层抛出晦涩的 TypeError，难以定位。
      本函数在 Python 层提前断言，使错误信息直接指向用户输入。

    Walpurgis 改写 (鲁迅拿法):
      上游用裸 assert，断言失败只有默认的 AssertionError 信息。
      本版保留 assert 语义，但在 WALPURGIS_DEBUG=1 时先打印诊断快照，
      使 CI 日志可读性从「一行 AssertionError」升级到「带上下文的证据链」。

    「凡事总须研究，才会明白。」——鲁迅《狂人日记》
    上游对 metadata 的校验散落在调用方，此处集中，调用方无需重复判断。

    Parameters
    ----------
    metadata : Optional[Dict[str, Union[str, Tuple[str, str, str]]]]
        异构图 metadata 字典，key 为属性名（str），
        value 为节点/边类型描述（str 或 3 元素 str 元组）。
        传入 None 时跳过校验（同构图路径）。

    Raises
    ------
    AssertionError
        若任何 key 不是 str，或 value 既非 str 也非 (str, str, str)。

    Examples
    --------
    >>> verify_metadata(None)  # ok，同构图
    >>> verify_metadata({"node_type": "paper"})  # ok
    >>> verify_metadata({"edge_type": ("author", "writes", "paper")})  # ok
    >>> verify_metadata({"bad": 42})  # AssertionError
    """
    if metadata is None:
        _dbg("verify_metadata", "metadata=None，跳过校验（同构图路径）")
        return

    _dbg(
        "verify_metadata",
        f"校验 metadata | keys={list(metadata.keys())} n_entries={len(metadata)}",
    )

    for k, v in metadata.items():
        # ── key 必须是 str ──────────────────────────────────────────────────
        assert isinstance(k, str), (
            f"Metadata keys must be strings. "
            f"Got key={k!r} (type={type(k).__name__})"
        )

        # ── value: str 或 (str, str, str) 元组 ─────────────────────────────
        if isinstance(v, tuple):
            # 3 元素同构 str 元组：代表 canonical edge type (src_type, rel, dst_type)
            assert len(v) == 3, (
                f"Metadata tuples must be of length 3. "
                f"Got key={k!r} tuple_len={len(v)} value={v!r}"
            )
            for i, elem in enumerate(v):
                assert isinstance(elem, str), (
                    f"Metadata tuple must be of type (str, str, str). "
                    f"Got key={k!r} tuple[{i}]={elem!r} (type={type(elem).__name__})"
                )
            _dbg(
                "verify_metadata",
                f"  ✓ key={k!r} → tuple ({v[0]!r}, {v[1]!r}, {v[2]!r})",
            )
        else:
            assert isinstance(v, str), (
                f"Metadata values must be strings or tuples of strings. "
                f"Got key={k!r} value={v!r} (type={type(v).__name__})"
            )
            _dbg("verify_metadata", f"  ✓ key={k!r} → str {v!r}")

    _dbg("verify_metadata", f"校验通过 | {len(metadata)} 个 metadata 条目全部合法")


# ─────────────────────────────────────────────────────────────────────────────
# HopIndexer — 封装 searchsorted hop 构建
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HopIndexer:
    """
    封装从 sampling_results.hop_id 构建 hop_starts 的 searchsorted 模式。

    上游在三个函数中各自重复:
        hops = torch.arange(sampling_results.hop_id.max() + 1, device="cuda")
        hops = torch.searchsorted(
            torch.as_tensor(sampling_results.hop_id, device="cuda"), hops
        )
    HopIndexer 提取为可复用对象，携带 hop_starts (hops) 和 num_hops。
    """
    hop_starts: "torch.Tensor"   # shape: (num_hops,), 每 hop 在 sampling_results 的起始行
    num_hops: int

    @classmethod
    def from_sampling_results(cls, sampling_results) -> "HopIndexer":
        max_hop = int(sampling_results.hop_id.max()) + 1
        hop_ids = torch.as_tensor(sampling_results.hop_id, device="cuda")
        arange = torch.arange(max_hop, device="cuda")
        hop_starts = torch.searchsorted(hop_ids, arange)
        _dbg("HopIndexer", "built", num_hops=max_hop, hop_starts=hop_starts.tolist())
        return cls(hop_starts=hop_starts, num_hops=max_hop)

    def hop_range(self, hop: int, total_rows: int) -> Tuple[int, int]:
        """返回第 hop 个 hop 在 sampling_results 中的 [start, end) 行范围。"""
        start = int(self.hop_starts[hop])
        end = int(self.hop_starts[hop + 1]) if hop < self.num_hops - 1 else total_rows
        return start, end


# ─────────────────────────────────────────────────────────────────────────────
# SamplerResultValidator — 封装 pylibcugraph 结果校验 + fallback
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplerResultValidator:
    """
    封装 pylibcugraph.negative_sampling 结果的数量校验和 fallback 填充。

    上游内联代码:
        if src_neg.numel() < num_neg:
            num_gen = num_neg - src_neg.numel()
            src_neg = torch.concat([src_neg, torch.randint(0, src_neg.max(), (num_gen,), ...)])
            dst_neg = torch.concat([dst_neg, torch.randint(0, dst_neg.max(), (num_gen,), ...)])
    SamplerResultValidator.validate_and_pad 给这段逻辑命名，并加断点观察 fallback 触发情况。
    """
    num_neg: int

    def validate_and_pad(
        self,
        src_neg: "torch.Tensor",
        dst_neg: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        actual = src_neg.numel()
        if actual >= self.num_neg:
            _dbg("SamplerResultValidator", "OK", num_neg=self.num_neg, actual=actual)
            return src_neg[: self.num_neg], dst_neg[: self.num_neg]

        # Fallback: pylibcugraph 生成数量不足（C API 边缘情况）
        num_gen = self.num_neg - actual
        _dbg(
            "SamplerResultValidator",
            "fallback triggered — padding with randint",
            expected=self.num_neg,
            actual=actual,
            num_gen=num_gen,
        )
        src_max = int(src_neg.max()) if actual > 0 else 1
        dst_max = int(dst_neg.max()) if actual > 0 else 1
        src_pad = torch.randint(0, src_max, (num_gen,), device="cuda", dtype=torch.int64)
        dst_pad = torch.randint(0, dst_max, (num_gen,), device="cuda", dtype=torch.int64)
        return torch.concat([src_neg, src_pad]), torch.concat([dst_neg, dst_pad])


# ─────────────────────────────────────────────────────────────────────────────
# 旧 Dask 路径：unique node helper（供 DaskGraphStore 路径使用）
# ─────────────────────────────────────────────────────────────────────────────

def _get_unique_nodes(
    sampling_results,
    graph_store,
    node_type: str,
    node_position: str,
) -> int:
    """
    从 sampling_results 中按节点类型计算唯一节点数（DaskGraphStore 路径）。

    Parameters
    ----------
    sampling_results : cudf.DataFrame
    graph_store : DaskGraphStore
    node_type : str
    node_position : str — "src" 或 "dst"
    """
    _dbg("_get_unique_nodes", f"node_type={node_type} position={node_position}")
    node_types, src_types, dst_types = graph_store._numeric_edge_types

    if node_position == "src":
        position_col = "majors"
        type_set = {i for i, t in enumerate(node_types) if t[0] == node_type}
    elif node_position == "dst":
        position_col = "minors"
        type_set = {i for i, t in enumerate(node_types) if t[2] == node_type}
    else:
        raise ValueError(f"node_position must be 'src' or 'dst', got {node_position!r}")

    if len(type_set) == 0:
        return cudf.Series([], dtype="int64")

    mask = sampling_results.edge_type.isin(type_set)
    return sampling_results[mask][position_col]


# ─────────────────────────────────────────────────────────────────────────────
# _sampler_output_from_sampling_results_homogeneous_coo
# ─────────────────────────────────────────────────────────────────────────────

def _sampler_output_from_sampling_results_homogeneous_coo(
    sampling_results,
    renumber_map,
    graph_store,
    data_index: Dict[Tuple[int, int], Dict[str, int]],
    batch_id: int,
    metadata: Sequence = None,
):
    """
    DaskGraphStore 路径: 从 COO 格式采样结果构建 HeteroSamplerOutput（同构图）。
    """
    if len(graph_store.edge_types) > 1 or len(graph_store.node_types) > 1:
        raise ValueError("Graph is heterogeneous")

    indexer = HopIndexer.from_sampling_results(sampling_results)

    node_type = graph_store.node_types[0]
    edge_type = graph_store.edge_types[0]

    num_nodes_per_hop_dict = {node_type: torch.zeros(indexer.num_hops + 1, dtype=torch.int64)}
    num_edges_per_hop_dict = {edge_type: torch.zeros(indexer.num_hops, dtype=torch.int64)}

    if renumber_map is None:
        raise ValueError("Renumbered input is expected for homogeneous graphs")

    noi_index = {node_type: torch.as_tensor(renumber_map, device="cuda")}

    row_dict = {edge_type: torch.as_tensor(sampling_results.majors, device="cuda")}
    col_dict = {edge_type: torch.as_tensor(sampling_results.minors, device="cuda")}

    num_nodes_per_hop_dict[node_type][0] = data_index[batch_id, 0]["src_max"] + 1

    total_rows = len(sampling_results)
    for hop in range(indexer.num_hops):
        if num_nodes_per_hop_dict[node_type][hop] > 0:
            max_id_hop = data_index[batch_id, hop]["dst_max"]
            max_id_prev = (
                data_index[batch_id, hop - 1]["dst_max"]
                if hop > 0
                else data_index[batch_id, 0]["src_max"]
            )
            delta = max_id_hop - max_id_prev if max_id_hop > max_id_prev else 0
            num_nodes_per_hop_dict[node_type][hop + 1] = delta

        hop_start, hop_end = indexer.hop_range(hop, total_rows)
        num_edges_per_hop_dict[edge_type][hop] = hop_end - hop_start

    _dbg(
        "_sampler_output_coo_homo",
        f"batch_id={batch_id}",
        num_nodes=num_nodes_per_hop_dict[node_type].tolist(),
        num_edges=num_edges_per_hop_dict[edge_type].tolist(),
    )

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes={k: t.tolist() for k, t in num_nodes_per_hop_dict.items()},
        num_sampled_edges={k: t.tolist() for k, t in num_edges_per_hop_dict.items()},
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _sampler_output_from_sampling_results_homogeneous_csr
# ─────────────────────────────────────────────────────────────────────────────

def _sampler_output_from_sampling_results_homogeneous_csr(
    major_offsets,
    minors,
    renumber_map,
    graph_store,
    label_hop_offsets,
    batch_id: int,
    metadata: Sequence = None,
):
    """
    DaskGraphStore 路径: 从 CSR/CSC 格式采样结果构建 HeteroSamplerOutput（同构图）。
    """
    if len(graph_store.edge_types) > 1 or len(graph_store.node_types) > 1:
        raise ValueError("Graph is heterogeneous")

    if renumber_map is None:
        raise ValueError("Renumbered input is expected for homogeneous graphs")

    node_type = graph_store.node_types[0]
    edge_type = graph_store.edge_types[0]

    major_offsets = major_offsets.clone() - major_offsets[0]
    label_hop_offsets = label_hop_offsets.clone() - label_hop_offsets[0]

    num_edges_per_hop_dict = {edge_type: major_offsets[label_hop_offsets].diff().tolist()}

    label_hop_offsets_cpu = label_hop_offsets.cpu()
    num_nodes_per_hop_dict = {
        node_type: torch.concat(
            [
                label_hop_offsets_cpu.diff(),
                (renumber_map.shape[0] - label_hop_offsets_cpu[-1]).reshape((1,)),
            ]
        ).tolist()
    }

    noi_index = {node_type: torch.as_tensor(renumber_map, device="cuda")}
    col_dict = {edge_type: major_offsets}
    row_dict = {edge_type: minors}

    _dbg(
        "_sampler_output_csr_homo",
        f"batch_id={batch_id}",
        num_nodes_len=len(num_nodes_per_hop_dict[node_type]),
        num_edges_len=len(num_edges_per_hop_dict[edge_type]),
    )

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes=num_nodes_per_hop_dict,
        num_sampled_edges=num_edges_per_hop_dict,
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _sampler_output_from_sampling_results_heterogeneous
# ─────────────────────────────────────────────────────────────────────────────

def _sampler_output_from_sampling_results_heterogeneous(
    sampling_results,
    renumber_map,
    graph_store,
    metadata: Sequence = None,
):
    """
    DaskGraphStore 路径: 从 COO 格式采样结果构建 HeteroSamplerOutput（异构图）。
    """
    indexer = HopIndexer.from_sampling_results(sampling_results)
    total_rows = len(sampling_results)

    num_nodes_per_hop_dict = {}
    num_edges_per_hop_dict = {}

    hop0_end = int(indexer.hop_starts[1]) if indexer.num_hops > 1 else total_rows
    sampling_results_hop_0 = sampling_results.iloc[0:hop0_end]

    for node_type in graph_store.node_types:
        num_unique = _get_unique_nodes(
            sampling_results_hop_0, graph_store, node_type, "src"
        ).nunique()
        if num_unique > 0:
            num_nodes_per_hop_dict[node_type] = torch.zeros(
                indexer.num_hops + 1, dtype=torch.int64
            )
            num_nodes_per_hop_dict[node_type][0] = num_unique

    if renumber_map is not None:
        raise ValueError(
            "Precomputing the renumber map is currently unsupported for heterogeneous graphs."
        )

    nodes_of_interest = (
        cudf.Series(
            torch.concat(
                [
                    torch.as_tensor(sampling_results_hop_0.majors, device="cuda"),
                    torch.as_tensor(sampling_results.minors, device="cuda"),
                ]
            ),
            name="nodes_of_interest",
        )
        .drop_duplicates()
        .sort_index()
    )

    noi_index = graph_store._get_vertex_groups_from_sample(
        torch.as_tensor(nodes_of_interest, device="cuda")
    )
    del nodes_of_interest

    row_dict, col_dict = graph_store._get_renumbered_edge_groups_from_sample(
        sampling_results, noi_index
    )

    for hop in range(indexer.num_hops):
        hop_start, hop_end = indexer.hop_range(hop, total_rows)
        sampling_results_to_hop = sampling_results.iloc[0:hop_end]

        for node_type in graph_store.node_types:
            unique_dst = _get_unique_nodes(
                sampling_results_to_hop, graph_store, node_type, "dst"
            )
            unique_src0 = _get_unique_nodes(
                sampling_results_hop_0, graph_store, node_type, "src"
            )
            num_unique = cudf.concat([unique_src0, unique_dst]).nunique()

            if num_unique > 0:
                if node_type not in num_nodes_per_hop_dict:
                    num_nodes_per_hop_dict[node_type] = torch.zeros(
                        indexer.num_hops + 1, dtype=torch.int64
                    )
                prev_sum = int(num_nodes_per_hop_dict[node_type][: hop + 1].sum(0))
                num_nodes_per_hop_dict[node_type][hop + 1] = num_unique - prev_sum

        numeric_etypes, counts = torch.unique(
            torch.as_tensor(
                sampling_results.iloc[hop_start:hop_end].edge_type, device="cuda"
            ),
            return_counts=True,
        )
        for num_etype, count in zip(numeric_etypes.tolist(), counts.tolist()):
            can_etype = graph_store.numeric_edge_type_to_canonical(num_etype)
            if can_etype not in num_edges_per_hop_dict:
                num_edges_per_hop_dict[can_etype] = torch.zeros(
                    indexer.num_hops, dtype=torch.int64
                )
            num_edges_per_hop_dict[can_etype][hop] = count

    _dbg(
        "_sampler_output_hetero",
        "done",
        node_types=list(num_nodes_per_hop_dict.keys()),
        edge_types=list(num_edges_per_hop_dict.keys()),
    )

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes={k: t.tolist() for k, t in num_nodes_per_hop_dict.items()},
        num_sampled_edges={k: t.tolist() for k, t in num_edges_per_hop_dict.items()},
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# filter_cugraph_pyg_store — 采样结果 → PyG Data 对象
# ─────────────────────────────────────────────────────────────────────────────

def filter_cugraph_pyg_store(
    feature_store,
    graph_store,
    node,
    row,
    col,
    edge,
    clx,
) -> "torch_geometric.data.Data":
    """
    将采样结果转换为 torch_geometric.data.Data，批量拉取特征。

    node: 节点 ID tensor (用于节点特征索引)
    edge: 边 ID tensor (用于边特征索引)
    group_name 为 tuple → 边特征，用 edge 索引；否则为节点特征，用 node 索引。
    """
    data = torch_geometric.data.Data()
    data.edge_index = torch.stack([row, col], dim=0)

    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        attr.index = edge if isinstance(attr.group_name, tuple) else node
        required_attrs.append(attr)
        data.num_nodes = attr.index.size(0)

    _dbg(
        "filter_cugraph_pyg_store",
        "fetching attrs",
        num_attrs=len(required_attrs),
        node_shape=node.shape if hasattr(node, "shape") else "?",
    )

    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.attr_name] = tensors[i]

    return data


# ─────────────────────────────────────────────────────────────────────────────
# neg_sample — 核心修复: 删除错误的分布式 all_reduce
# ─────────────────────────────────────────────────────────────────────────────

def neg_sample(
    graph_store,
    seed_src: "torch.Tensor",
    seed_dst: "torch.Tensor",
    input_type,
    batch_size: int,
    neg_sampling: "torch_geometric.sampler.NegativeSampling",
    time: "torch.Tensor",
    node_time: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    为链路预测生成负样本。

    dd543dc 核心修复: 删除错误的分布式 all_reduce SUM。
    旧代码将各 rank 的 num_neg 求和后传给 pylibcugraph，
    导致每 rank 生成 world_size 倍的负样本（正负比失衡，静默影响 AUC）。
    修复: 每 rank 独立使用本地 num_neg，不进行跨 rank 对齐。

    Parameters
    ----------
    graph_store : GraphStore
    seed_src, seed_dst : 正样本边两端节点
    input_type : 边类型（None 或 canonical tuple）
    batch_size : 批大小
    neg_sampling : PyG NegativeSampling 配置
    time, node_time : 时序采样时间戳（当前仅 None 被支持）
    """
    try:
        # PyG 2.5 兼容
        src_weight = neg_sampling.src_weight
        dst_weight = neg_sampling.dst_weight
    except AttributeError:
        src_weight = neg_sampling.weight
        dst_weight = neg_sampling.weight

    unweighted = src_weight is None and dst_weight is None

    # 至少每 batch 一个负样本
    num_neg = max(
        int(ceil(neg_sampling.amount * seed_src.numel())),
        int(ceil(seed_src.numel() / batch_size)),
    )

    _dbg(
        "neg_sample",
        "start",
        num_neg=num_neg,
        seed_src_n=seed_src.numel(),
        unweighted=unweighted,
        input_type=str(input_type),
    )

    if node_time is None:
        result_dict = pylibcugraph.negative_sampling(
            graph_store._resource_handle,
            graph_store._graph,
            num_neg,  # 修复: 不再用 num_neg_global (all_reduce SUM 后的膨胀值)
            vertices=(
                None
                if unweighted
                else cupy.arange(src_weight.numel(), dtype="int64")
            ),
            src_bias=None if src_weight is None else cupy.asarray(src_weight),
            dst_bias=None if dst_weight is None else cupy.asarray(dst_weight),
            remove_duplicates=False,
            remove_false_negatives=False,
            exact_number_of_samples=True,
            do_expensive_check=False,
        )

        src_neg = torch.as_tensor(result_dict["sources"], device="cuda")
        dst_neg = torch.as_tensor(result_dict["destinations"], device="cuda")

        # 校验数量 + fallback 填充（C API 边缘情况）
        validator = SamplerResultValidator(num_neg=num_neg)
        src_neg, dst_neg = validator.validate_and_pad(src_neg, dst_neg)

        _dbg("neg_sample", "done", src_neg_shape=src_neg.shape, dst_neg_shape=dst_neg.shape)
        return src_neg, dst_neg

    raise NotImplementedError(
        "Temporal negative sampling is currently unimplemented in Walpurgis/cuGraph-PyG"
    )


# ─────────────────────────────────────────────────────────────────────────────
# neg_cat — 将正负样本按 batch 交错拼接
# ─────────────────────────────────────────────────────────────────────────────

def neg_cat(
    seed_pos: "torch.Tensor",
    seed_neg: "torch.Tensor",
    pos_batch_size: int,
) -> Tuple["torch.Tensor", int]:
    """
    将正负样本按 batch 对齐交错拼接。

    返回 (拼接后的 tensor, neg_batch_size)。
    neg_batch_size 是每个 batch 中负样本的数量，用于后续分离正负标签。
    """
    num_seeds = seed_pos.numel()
    num_batches = int(ceil(num_seeds / pos_batch_size))
    neg_batch_size = int(ceil(seed_neg.numel() / num_batches))

    _dbg(
        "neg_cat",
        "interleaving",
        num_batches=num_batches,
        pos_batch_size=pos_batch_size,
        neg_batch_size=neg_batch_size,
    )

    batch_pos_offsets = torch.full((num_batches,), pos_batch_size).cumsum(-1)[:-1]
    seed_pos_splits = torch.tensor_split(seed_pos, batch_pos_offsets)

    batch_neg_offsets = torch.full((num_batches,), neg_batch_size).cumsum(-1)[:-1]
    seed_neg_splits = torch.tensor_split(seed_neg, batch_neg_offsets)

    result = torch.concatenate(
        [torch.concatenate(s) for s in zip(seed_pos_splits, seed_neg_splits)]
    )
    _dbg("neg_cat", "done", result_shape=result.shape)
    return result, neg_batch_size
