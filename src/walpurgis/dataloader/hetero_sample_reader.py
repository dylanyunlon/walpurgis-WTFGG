# Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# Licensed under the Apache License, Version 2.0.
#
# 迁移来源: cugraph-gnn commit 24e91be
# 原标题: [BUG] Specify Input Type and Assign Output to Correct Type
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 修了两个在异构图采样中潜伏已久的虫子。
# 第一个虫子：input_type 在采样之后就再找不着了，
# 最后往 num_sampled_nodes 和 num_sampled_edges 里填数据，
# 全填到了错误的类型桶子里——偏偏程序还能跑，只是悄悄出错。
# 第二个虫子：edge_inverse 没有做 de-offset，
# 返回的节点编号是全局 offset 过的，不是 PyG 期望的局部 ID——
# 然而模型照样能收敛，只是精度悄悄下去了。
# 两个虫子都属于"沉默的杀手"型，不抛异常，只腐蚀结果。

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# 数据类：对应上游 raw_sample_data["input_type"] 的语义
# ---------------------------------------------------------------------------

@dataclass
class InputTypeSpec:
    """
    封装 heterogeneous 采样中的 input_type 语义。

    上游原代码中 input_type 可以是:
    - None          → 错误，24e91be 之前会到循环末才抛 ValueError
    - str           → 节点采样，值是节点类型名 (pyg_can_etype[2] 即 dst node type)
    - Tuple[str,str,str] → 边采样，值是 canonical edge type (src, rel, dst)

    上游 bug 根因:
    - 24e91be 之前 input_type 在循环里被"发现"而非"传入"：
      通过检查 col[pyg_can_etype][:hop0].numel() > 0 来猜测哪个边类型是 input，
      但这个启发式在某些采样结果里会猜错（空边、共享节点类型）。
    - 修复后：input_type 作为 metadata 从 BaseSampler 传入，
      _decode 直接从 raw_sample_data["input_type"] 读取，不再猜测。
    """

    raw: Union[str, Tuple[str, str, str]]  # 不允许 None，构造前已校验

    @property
    def is_node_input(self) -> bool:
        """节点采样: input_type 是 str（节点类型名）"""
        return isinstance(self.raw, str)

    @property
    def is_edge_input(self) -> bool:
        """边采样: input_type 是 (src, rel, dst) tuple"""
        return isinstance(self.raw, tuple)

    def matches_edge_type(self, pyg_can_etype: Tuple[str, str, str]) -> bool:
        """
        检查是否与给定的 canonical edge type 匹配。
        - 边采样: 直接比较 tuple
        - 节点采样: 比较 dst node type (pyg_can_etype[2])
        """
        if self.is_edge_input:
            return self.raw == pyg_can_etype
        else:
            return self.raw == pyg_can_etype[2]

    def validate(self):
        if self.raw is None:
            raise ValueError(
                "[Walpurgis] InputTypeSpec.raw 不能为 None。"
                "确认 BaseSampler 已将 index.input_type 作为 metadata 传入采样器。"
            )
        if not isinstance(self.raw, (str, tuple)):
            raise TypeError(
                f"[Walpurgis] input_type 期望 str 或 Tuple[str,str,str]，"
                f"实际得到 {type(self.raw).__name__}: {self.raw!r}"
            )
        if isinstance(self.raw, tuple) and len(self.raw) != 3:
            raise ValueError(
                f"[Walpurgis] edge input_type tuple 长度必须为 3，"
                f"实际: {self.raw!r}"
            )
        if _WDBG:
            print(
                f"[WALPURGIS_DEBUG] InputTypeSpec.validate OK "
                f"raw={self.raw!r} is_node={self.is_node_input}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# 核心函数：对应上游 __decode_coo 里"确定 integer_input_type"的逻辑
# ---------------------------------------------------------------------------

def resolve_integer_input_type(
    input_type_spec: InputTypeSpec,
    edge_types: List[Tuple[str, str, str]],
    src_types,  # torch.Tensor: 每个 edge_type 对应的 src vertex type index
    dst_types,  # torch.Tensor: 每个 edge_type 对应的 dst vertex type index
) -> int:
    """
    将 input_type（字符串或 tuple）映射到整数 index。

    上游逻辑（24e91be 之后）:
    - 如果 input_type == pyg_can_etype (tuple 匹配): integer_input_type = etype (边类型 index)
    - 如果 input_type == pyg_can_etype[2] (str 匹配 dst node): integer_input_type = src_types[etype]

    24e91be 之前的 bug:
    - integer_input_type 根本不存在，上游把 input_type 当字符串或 tuple 直接传给
      SamplerOutput 的 metadata[0]，但 SampleIterator._next_sample 里:
          input_type, input_id = next_sample.metadata[0]
      这行要求 metadata[0] 是 (input_type, input_id) 形式，
      结果 input_type 被赋值成了错误类型对象，后续 data[input_type] 索引偏了。

    Returns
    -------
    int: 对应 edge_types 列表中的 index（边采样）
         或 src_types[etype] 的值（节点采样）
    """
    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] resolve_integer_input_type 入口 "
            f"input_type={input_type_spec.raw!r} "
            f"num_edge_types={len(edge_types)}",
            file=sys.stderr,
        )

    for etype_idx, pyg_can_etype in enumerate(edge_types):
        if input_type_spec.is_edge_input and input_type_spec.raw == pyg_can_etype:
            # 边采样：返回边类型的整数 index
            if _WDBG:
                print(
                    f"[WALPURGIS_DEBUG] 边采样匹配 etype_idx={etype_idx} "
                    f"pyg_can_etype={pyg_can_etype}",
                    file=sys.stderr,
                )
            return etype_idx

        if input_type_spec.is_node_input and input_type_spec.raw == pyg_can_etype[2]:
            # 节点采样：返回该边类型的 src vertex type index
            src_type_int = int(src_types[etype_idx].item())
            if _WDBG:
                print(
                    f"[WALPURGIS_DEBUG] 节点采样匹配 etype_idx={etype_idx} "
                    f"pyg_can_etype={pyg_can_etype} "
                    f"src_type_int={src_type_int}",
                    file=sys.stderr,
                )
            return src_type_int

    # 上游原始报错: "Input type did not match any edge type!"
    raise ValueError(
        f"[Walpurgis] input_type {input_type_spec.raw!r} 与所有 edge_type 均不匹配。\n"
        f"已知 edge_types: {edge_types}\n"
        f"检查 BaseSampler 传入的 index.input_type 是否与图中 edge/node type 名称一致。"
    )


# ---------------------------------------------------------------------------
# 核心函数：对应上游 __decode_coo 里 edge_inverse de-offset 逻辑
# ---------------------------------------------------------------------------

def deoffset_edge_inverse(
    edge_inverse,  # torch.Tensor, shape [2*N] 或已 view 成 [2, N]
    input_type_spec: InputTypeSpec,
    integer_input_type: int,
    src_types,       # torch.Tensor
    dst_types,       # torch.Tensor
    vertex_offsets,  # torch.Tensor: 每个 vertex type 的全局 offset
):
    """
    对 edge_inverse 做 de-offset，返回局部 vertex ID。

    上游 bug（24e91be 之前）:
    - edge_inverse 被直接 view(2, -1) 后塞入 metadata，
      但其中的节点编号是带全局 vertex offset 的（cuGraph sampler 输出格式），
      PyG 期望的是 per-type 局部 ID（从 0 开始）。
    - 这导致 edge_label_index 里的节点 ID 偏移整个图的顶点数量，
      训练时 embedding lookup 会超出范围或命中错误节点。

    修复逻辑（24e91be）:
    - 边采样: 分别对 src/dst 行减去对应类型的 vertex_offset
    - 节点采样: input_type 是 str，不应该有 edge_inverse，直接 raise

    Returns
    -------
    torch.Tensor: shape [2, N]，已 de-offset 的 edge_inverse
    """
    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] deoffset_edge_inverse 入口 "
            f"edge_inverse.shape={tuple(edge_inverse.shape)} "
            f"input_type={input_type_spec.raw!r} "
            f"integer_input_type={integer_input_type}",
            file=sys.stderr,
        )

    # 统一 reshape 成 [2, N]
    ei = edge_inverse.view(2, -1)

    if input_type_spec.is_node_input:
        # 节点采样不应该走到这里（没有 edge_inverse 语义）
        # 上游原文: raise ValueError("Input type should be a tuple for edge input.")
        raise ValueError(
            "[Walpurgis] deoffset_edge_inverse: "
            "节点采样（str input_type）不应存在 edge_inverse。\n"
            f"实际 input_type={input_type_spec.raw!r}。\n"
            "若调用方传入了 edge_inverse，检查 LinkNeighborLoader 配置是否误传 edge 输入。"
        )

    # 边采样: integer_input_type 是 etype index (int)
    src_offset = int(vertex_offsets[src_types[integer_input_type]].item())
    dst_offset = int(vertex_offsets[dst_types[integer_input_type]].item())

    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] deoffset: "
            f"src_offset={src_offset} dst_offset={dst_offset} "
            f"ei[0].min={int(ei[0].min().item())} ei[0].max={int(ei[0].max().item())} "
            f"ei[1].min={int(ei[1].min().item())} ei[1].max={int(ei[1].max().item())}",
            file=sys.stderr,
        )

    ei[0] -= src_offset
    ei[1] -= dst_offset

    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] deoffset 完成: "
            f"ei[0].min={int(ei[0].min().item())} ei[0].max={int(ei[0].max().item())} "
            f"ei[1].min={int(ei[1].min().item())} ei[1].max={int(ei[1].max().item())}",
            file=sys.stderr,
        )

    return ei


# ---------------------------------------------------------------------------
# 核心函数：num_sampled_nodes 填充逻辑（对应上游 __decode_coo 中的第一个 bug 修复）
# ---------------------------------------------------------------------------

def update_num_sampled_nodes_for_input(
    input_type_spec: InputTypeSpec,
    etype_idx: int,
    num_sampled_nodes: list,
    col,           # Dict[Tuple, torch.Tensor]
    row,           # Dict[Tuple, torch.Tensor]
    pyg_can_etype: Tuple[str, str, str],
    num_sampled_edges: dict,
    src_types,
    dst_types,
):
    """
    为 seed 层（hop=0）的 num_sampled_nodes 填入正确的节点数量。

    上游 bug（24e91be 之前）:
    - 旧代码只更新 dst_types 的 num_sampled_nodes[0]，遗漏了 src_types。
    - 对于边采样（edge_inverse case），src 侧的种子节点数量从未被正确记录，
      导致 HeteroData.num_sampled_nodes 对 src 类型返回 0，
      下游 GNN 层在计算 src embedding 时使用了错误的 batch 大小。

    修复逻辑（24e91be）:
    - 边采样: 同时更新 dst 和 src 的 num_sampled_nodes[0]
    - 节点采样: 只更新 dst（原逻辑），但现在受 numel() > 0 guard 保护
    """
    import torch

    hop0_edges = num_sampled_edges[pyg_can_etype][0]

    if input_type_spec.is_edge_input and input_type_spec.raw == pyg_can_etype:
        # 边采样路径
        ux = col[pyg_can_etype][:hop0_edges]
        uy = row[pyg_can_etype][:hop0_edges]

        if _WDBG:
            print(
                f"[WALPURGIS_DEBUG] update_num_sampled_nodes 边采样 "
                f"etype={pyg_can_etype} hop0_edges={int(hop0_edges.item())} "
                f"ux.shape={tuple(ux.shape)} uy.shape={tuple(uy.shape)}",
                file=sys.stderr,
            )

        # 更新 dst
        num_sampled_nodes[dst_types[etype_idx]][0] = torch.max(
            num_sampled_nodes[dst_types[etype_idx]][0],
            (ux.max() + 1).reshape((1,)),
        )
        # 更新 src（24e91be 新增，修复遗漏）
        num_sampled_nodes[src_types[etype_idx]][0] = torch.max(
            num_sampled_nodes[src_types[etype_idx]][0],
            (uy.max() + 1).reshape((1,)),
        )

    elif (
        input_type_spec.is_node_input
        and input_type_spec.raw == pyg_can_etype[2]
    ):
        # 节点采样路径
        ux = col[pyg_can_etype][:hop0_edges]

        if _WDBG:
            print(
                f"[WALPURGIS_DEBUG] update_num_sampled_nodes 节点采样 "
                f"etype={pyg_can_etype} hop0_edges={int(hop0_edges.item())} "
                f"ux.shape={tuple(ux.shape)} ux.numel={ux.numel()}",
                file=sys.stderr,
            )

        if ux.numel() > 0:
            num_sampled_nodes[dst_types[etype_idx]][0] = torch.max(
                num_sampled_nodes[dst_types[etype_idx]][0],
                (ux.max() + 1).reshape((1,)),
            )


# ---------------------------------------------------------------------------
# 核心函数：BaseSampler 端 —— 将 input_type 打包进 metadata
# ---------------------------------------------------------------------------

def build_sampler_metadata(
    input_type: Optional[Union[str, Tuple[str, str, str]]]
) -> Optional[Dict[str, Union[str, Tuple[str, str, str]]]]:
    """
    将 index.input_type 包装成采样器能接收的 metadata dict。

    上游 24e91be 在 BaseSampler.sample_from_nodes 和 sample_from_edges
    中均新增了此逻辑：

        metadata = (
            {"input_type": index.input_type}
            if index.input_type is not None
            else None
        )

    这是整个 fix 的"源头"——没有这一步，_decode 里读 raw_sample_data["input_type"]
    就永远是 None，两个下游 bug 就无从修复。

    Walpurgis 迁移：提取为可独立测试的函数。
    """
    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] build_sampler_metadata input_type={input_type!r}",
            file=sys.stderr,
        )

    if input_type is None:
        return None

    metadata = {"input_type": input_type}

    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] build_sampler_metadata → {metadata}",
            file=sys.stderr,
        )

    return metadata


# ---------------------------------------------------------------------------
# 校验函数：_decode 入口的类型注解校验（对应上游类型签名升级）
# ---------------------------------------------------------------------------

def validate_raw_sample_data_input_type(
    raw_sample_data: Dict,
    context: str = "_decode",
):
    """
    对应上游将 _decode / __decode_csc / __decode_coo 的函数签名从:
        Dict[str, "torch.Tensor"]
    升级为:
        Dict[str, Union["torch.Tensor", str, Tuple[str, str, str]]]

    上游只升级了类型注解，运行时无校验。
    Walpurgis 迁移：加显式运行时校验 + 调试 print。

    24e91be 之前 raw_sample_data["input_type"] 根本不存在于 dict 中
    （采样器从未写入），所以取到的是 dict.get() 的 None，掩盖了整个传递链断裂。
    """
    input_type = raw_sample_data.get("input_type")

    if _WDBG:
        print(
            f"[WALPURGIS_DEBUG] validate_raw_sample_data [{context}] "
            f"input_type={input_type!r} "
            f"keys={list(raw_sample_data.keys())}",
            file=sys.stderr,
        )

    if input_type is None:
        raise ValueError(
            f"[Walpurgis:{context}] raw_sample_data 中缺少 'input_type'。\n"
            "确认 BaseSampler.sample_from_nodes / sample_from_edges "
            "已将 metadata={'input_type': ...} 传递给采样器。\n"
            "这是 24e91be 修复的关键传递链——若此处失败，检查 BaseSampler 版本。"
        )

    if not isinstance(input_type, (str, tuple)):
        raise TypeError(
            f"[Walpurgis:{context}] input_type 应为 str 或 Tuple[str,str,str]，"
            f"得到 {type(input_type).__name__}: {input_type!r}"
        )

    return InputTypeSpec(raw=input_type)


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ["WALPURGIS_DEBUG"] = "1"

    print("=== 自测 hetero_sample_reader.py (migrate 24e91be) ===\n")

    # --- 测试 1: InputTypeSpec 节点采样 ---
    spec_node = InputTypeSpec(raw="paper")
    spec_node.validate()
    assert spec_node.is_node_input
    assert not spec_node.is_edge_input
    # "paper" 匹配 dst node type = pyg_can_etype[2]
    assert spec_node.matches_edge_type(("author", "writes", "paper"))   # dst="paper" ✓
    assert spec_node.matches_edge_type(("paper", "cites", "paper"))     # dst="paper" ✓
    assert not spec_node.matches_edge_type(("paper", "cites", "author"))  # dst="author" ✗
    print("[OK] 测试1: InputTypeSpec 节点采样")

    # --- 测试 2: InputTypeSpec 边采样 ---
    spec_edge = InputTypeSpec(raw=("author", "writes", "paper"))
    spec_edge.validate()
    assert spec_edge.is_edge_input
    assert not spec_edge.is_node_input
    assert spec_edge.matches_edge_type(("author", "writes", "paper"))
    assert not spec_edge.matches_edge_type(("paper", "cites", "paper"))
    print("[OK] 测试2: InputTypeSpec 边采样")

    # --- 测试 3: build_sampler_metadata ---
    m = build_sampler_metadata("paper")
    assert m == {"input_type": "paper"}
    m2 = build_sampler_metadata(None)
    assert m2 is None
    print("[OK] 测试3: build_sampler_metadata")

    # --- 测试 4: validate_raw_sample_data_input_type 缺 key ---
    try:
        validate_raw_sample_data_input_type({}, "_decode")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "缺少 'input_type'" in str(e)
        print("[OK] 测试4: 缺少 input_type 报错正确")

    # --- 测试 5: validate_raw_sample_data_input_type 正常 ---
    spec = validate_raw_sample_data_input_type(
        {"input_type": ("author", "writes", "paper")}, "_decode"
    )
    assert isinstance(spec, InputTypeSpec)
    print("[OK] 测试5: validate_raw_sample_data_input_type 正常路径")

    # --- 测试 6: resolve_integer_input_type 边采样 ---
    import torch

    edge_types = [
        ("paper", "cites", "paper"),
        ("author", "writes", "paper"),
    ]
    src_types = torch.tensor([0, 1])  # paper=0, author=1
    dst_types = torch.tensor([0, 0])  # paper=0, paper=0

    spec_e = InputTypeSpec(raw=("author", "writes", "paper"))
    idx = resolve_integer_input_type(spec_e, edge_types, src_types, dst_types)
    assert idx == 1, f"期望 1 得到 {idx}"
    print("[OK] 测试6: resolve_integer_input_type 边采样")

    # --- 测试 7: resolve_integer_input_type 节点采样 ---
    spec_n = InputTypeSpec(raw="paper")
    idx_n = resolve_integer_input_type(spec_n, edge_types, src_types, dst_types)
    # paper 作为 dst 出现在 (paper,cites,paper)[2] 和 (author,writes,paper)[2]
    # 先匹配 (paper,cites,paper)，返回 src_types[0] = 0
    assert idx_n == 0, f"期望 0 得到 {idx_n}"
    print("[OK] 测试7: resolve_integer_input_type 节点采样")

    # --- 测试 8: deoffset_edge_inverse ---
    vertex_offsets = torch.tensor([0, 100, 200])  # paper=0, author=100(unused), dst_paper=...
    # 模拟 edge_inverse: src 节点编号 100+, dst 节点编号 0+（没有 offset，paper 从 0 开始）
    ei_raw = torch.tensor([100, 101, 102, 0, 1, 2])  # src 行有 offset=100

    # 边采样，integer_input_type=1 (author,writes,paper): src=author(offset=100), dst=paper(offset=0)
    src_types2 = torch.tensor([0, 1])  # etype 0=paper, etype 1=author
    dst_types2 = torch.tensor([0, 0])
    vertex_offsets2 = torch.tensor([0, 100])  # paper=0, author=100

    spec_e2 = InputTypeSpec(raw=("author", "writes", "paper"))
    ei_out = deoffset_edge_inverse(
        ei_raw, spec_e2,
        integer_input_type=1,
        src_types=src_types2,
        dst_types=dst_types2,
        vertex_offsets=vertex_offsets2,
    )
    assert ei_out[0].tolist() == [0, 1, 2], f"src de-offset 失败: {ei_out[0].tolist()}"
    assert ei_out[1].tolist() == [0, 1, 2], f"dst de-offset 失败: {ei_out[1].tolist()}"
    print("[OK] 测试8: deoffset_edge_inverse")

    # --- 测试 9: 节点采样 deoffset 应该抛 ValueError ---
    spec_n2 = InputTypeSpec(raw="paper")
    try:
        deoffset_edge_inverse(
            ei_raw, spec_n2, 0, src_types2, dst_types2, vertex_offsets2
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "节点采样" in str(e)
        print("[OK] 测试9: 节点采样 deoffset 正确抛错")

    print("\n=== 全部自测通过 ===")
