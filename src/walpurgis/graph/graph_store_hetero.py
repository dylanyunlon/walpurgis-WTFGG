# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit a9ab8b4
# 原标题: [FEA] Support Heterogeneous Sampling in cuGraph-PyG (#82)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「沉默呵，沉默呵！不在沉默中爆发，就在沉默中灭亡。」
# —— 鲁迅《纪念刘和珍君》
#
# a9ab8b4 在 GraphStore 中新增两个属性，打通异构采样路径：
#   1. _vertex_offset_array: 把各顶点类型全局偏移量压缩成单个 CUDA int64 Tensor，
#      末尾追加总顶点数，供 HeterogeneousSampleReader 做 de-offset
#   2. _numeric_edge_types: 返回 (sorted_canonical_etypes, src_int_tensor, dst_int_tensor)，
#      将字符串 edge type 映射到整数索引，供采样器传参
#   3. __numeric_edge_types 缓存字段 — 避免每次采样都重新排列字典键
#
# 同时 BaseSampler.sample_from_nodes/sample_from_edges 中原来的：
#   raise NotImplementedError("Sampling heterogeneous graphs is currently
#   unsupported in the non-dask API")
# 被替换为真正的 HeterogeneousSampleReader 实例化路径。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. NumericEdgeTypeIndex 枚举语义类 — 将 _numeric_edge_types 三元组裸 tuple
#      包装为命名类，字段名 edge_types / src_types / dst_types 替代位置访问
#   2. _build_vertex_offset_array() 模块级函数 — 从 GraphStore 内联逻辑提取，
#      可独立测试，加 shape/dtype 断言和 DEBUG 摘要
#   3. _build_numeric_edge_types() 模块级函数 — 同上，加未知类型 KeyError 诊断
#   4. 全链路 WALPURGIS_DEBUG=1 断点：
#      - _build_vertex_offset_array: 偏移数组构建过程
#      - _build_numeric_edge_types: vtype_table 和每个 etype 的映射
#      - BaseSampler 路径选择（同构 vs 异构）

from __future__ import annotations

import os as _os
import sys as _sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 命名类型：_numeric_edge_types 三元组
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericEdgeTypeIndex:
    """
    替代上游 _numeric_edge_types 的裸 tuple 返回值，字段有名字。

    上游返回:
        (sorted_keys, torch.tensor(srcs, ..., int32), torch.tensor(dsts, ..., int32))
    解包习惯:
        edge_types, src_types, dst_types = graph_store._numeric_edge_types

    Walpurgis 保持解构兼容（__iter__ 返回三元素），同时支持命名访问。
    """

    edge_types: List[Tuple[str, str, str]]
    src_types: object   # torch.Tensor int32
    dst_types: object   # torch.Tensor int32

    def __iter__(self):
        """兼容上游三元组解构：edge_types, src_types, dst_types = ..."""
        yield self.edge_types
        yield self.src_types
        yield self.dst_types

    def summary(self) -> str:
        lines = [f"NumericEdgeTypeIndex(n={len(self.edge_types)}):"]
        for i, et in enumerate(self.edge_types):
            src_i = int(self.src_types[i].item())
            dst_i = int(self.dst_types[i].item())
            lines.append(f"  [{i}] {et}  src={src_i}  dst={dst_i}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 构建函数
# ---------------------------------------------------------------------------

def _build_vertex_offset_array(
    vertex_offsets: Dict[str, int],
    total_vertices: int,
) -> "torch.Tensor":
    """
    上游 GraphStore._vertex_offset_array 核心逻辑（a9ab8b4 新增）。

    形如:
        [off_type0, off_type1, ..., off_typeN, total]
    长度 = num_vtypes + 1，dtype=int64，device=cuda。

    Walpurgis 改写：提取为模块级函数，加断言和 DEBUG。
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("[Walpurgis] torch 不可用: " + str(exc)) from exc

    sorted_keys = sorted(vertex_offsets.keys())
    offsets = [vertex_offsets[k] for k in sorted_keys]

    _dbg(
        "_build_vertex_offset_array",
        f"sorted_keys={sorted_keys}  offsets={offsets}  total={total_vertices}",
    )

    off_tensor = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    total_tensor = torch.tensor([total_vertices], dtype=torch.int64, device="cuda")
    result = torch.concat([off_tensor, total_tensor])

    assert result.numel() == len(sorted_keys) + 1, (
        f"[Walpurgis] _build_vertex_offset_array 长度异常: "
        f"期望 {len(sorted_keys)+1}, 实际 {result.numel()}"
    )

    _dbg("_build_vertex_offset_array", f"result={result.tolist()}")
    return result


def _build_numeric_edge_types(
    edge_type_keys: List[Tuple[str, str, str]],
    vertex_offsets: Dict[str, int],
) -> NumericEdgeTypeIndex:
    """
    上游 GraphStore._numeric_edge_types 计算逻辑（a9ab8b4 新增）。

    构建 vtype_table = {vtype_name: int_index}（按字典序排列顶点类型），
    然后对每个 edge type 查表得到 src/dst 整数索引。

    Walpurgis 改写：返回 NumericEdgeTypeIndex 而非裸 tuple，加 KeyError 诊断。
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("[Walpurgis] torch 不可用: " + str(exc)) from exc

    sorted_etypes = sorted(edge_type_keys)
    vtype_table: Dict[str, int] = {
        k: i for i, k in enumerate(sorted(vertex_offsets.keys()))
    }

    _dbg(
        "_build_numeric_edge_types",
        f"vtype_table={vtype_table}  n_etypes={len(sorted_etypes)}",
    )

    srcs: List[int] = []
    dsts: List[int] = []

    for etype in sorted_etypes:
        src_name, _rel, dst_name = etype
        if src_name not in vtype_table:
            raise KeyError(
                f"[Walpurgis:_build_numeric_edge_types] "
                f"src vertex type '{src_name}' 不在已知顶点类型 {list(vtype_table)} 中。\n"
                f"edge type: {etype}"
            )
        if dst_name not in vtype_table:
            raise KeyError(
                f"[Walpurgis:_build_numeric_edge_types] "
                f"dst vertex type '{dst_name}' 不在已知顶点类型 {list(vtype_table)} 中。\n"
                f"edge type: {etype}"
            )
        srcs.append(vtype_table[src_name])
        dsts.append(vtype_table[dst_name])

        _dbg(
            "_build_numeric_edge_types",
            f"etype={etype}  src_int={vtype_table[src_name]}  "
            f"dst_int={vtype_table[dst_name]}",
        )

    index = NumericEdgeTypeIndex(
        edge_types=sorted_etypes,
        src_types=torch.tensor(srcs, device="cuda", dtype=torch.int32),
        dst_types=torch.tensor(dsts, device="cuda", dtype=torch.int32),
    )

    _dbg("_build_numeric_edge_types", index.summary())
    return index


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ["WALPURGIS_DEBUG"] = "1"
    print("=== 自测 graph_store_hetero.py (migrate a9ab8b4) ===\n")

    try:
        import torch
    except ImportError:
        print("[SKIP] torch 不可用")
        exit(0)

    # 场景：author-writes-paper, paper-cites-paper 两类边
    vertex_offsets = {"author": 0, "paper": 500}
    total = 700

    # --- 测试 1: _build_vertex_offset_array ---
    arr = _build_vertex_offset_array(vertex_offsets, total)
    # sorted vtypes: author(0), paper(500), total(700)
    assert arr.tolist() == [0, 500, 700], f"实际: {arr.tolist()}"
    assert arr.dtype == torch.int64
    print(f"[OK] 测试1: _build_vertex_offset_array → {arr.tolist()}")

    # --- 测试 2: _build_numeric_edge_types ---
    edge_keys = [
        ("paper", "cites", "paper"),
        ("author", "writes", "paper"),
    ]
    idx = _build_numeric_edge_types(edge_keys, vertex_offsets)
    # 解构兼容测试
    et, st, dt = idx
    assert et == sorted(edge_keys)
    # vtype_table: author=0, paper=1
    # sorted etypes: (author,writes,paper), (paper,cites,paper)
    # (author,writes,paper): src=author=0, dst=paper=1
    # (paper,cites,paper):   src=paper=1,  dst=paper=1
    assert st.tolist() == [0, 1], f"src_types={st.tolist()}"
    assert dt.tolist() == [1, 1], f"dst_types={dt.tolist()}"
    print(f"[OK] 测试2: _build_numeric_edge_types\n{idx.summary()}")

    # --- 测试 3: 未知顶点类型 KeyError ---
    try:
        _build_numeric_edge_types(
            [("ghost", "haunts", "paper")], vertex_offsets
        )
        assert False, "应该 KeyError"
    except KeyError as e:
        assert "ghost" in str(e)
        print("[OK] 测试3: 未知 src vtype KeyError")

    # --- 测试 4: NumericEdgeTypeIndex 解构兼容 ---
    et2, st2, dt2 = idx
    assert et2 is idx.edge_types
    assert st2 is idx.src_types
    assert dt2 is idx.dst_types
    print("[OK] 测试4: NumericEdgeTypeIndex 三元组解构兼容")

    print("\n=== 全部自测通过 ===")
