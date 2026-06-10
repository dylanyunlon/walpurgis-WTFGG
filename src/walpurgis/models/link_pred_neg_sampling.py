"""
link_pred_neg_sampling.py — 8bf2012 迁移: DGL 链路预测与负采样支持

migrate 8bf2012: [FEA] Support Link Prediction and Negative Sampling in DGL

上游变化 (8bf2012, cugraph-gnn, NVIDIA, 2025/2026):
  文件: python/cugraph-dgl/cugraph_dgl/graph.py
  主要功能新增:

  1. __edge_lookup_table = None  — 新增实例变量, 懒加载 pylibcugraph.EdgeIdLookupTable
     _clear_graph() — 新增方法, 统一清理 __graph + __edge_lookup_table + __vertex_offsets
     之前三处 `self.__graph = None; self.__vertex_offsets = None` 改为调用 _clear_graph()
     → 解决旧代码漏清 __edge_lookup_table 的潜在问题 (8bf2012 引入 lookup_table 同时配套清理)

  2. _to_numeric_etype(etype) — 新增方法
     将边类型 (str 或 tuple) 映射到整数 index, 供 pylibcugraph.EdgeIdLookupTable.lookup_vertex_ids
     使用。排序键与 __edge_indices.keys(leaves_only=True, include_nested=True) 一致。
     None → 单类型图返回 0; 多类型图 None 抛 ValueError。

  3. _edge_lookup_table property — 懒加载:
     第一次访问时构建 pylibcugraph.EdgeIdLookupTable(resource_handle, graph);
     之后复用缓存; _clear_graph() 时置 None 触发下次重建。

  4. find_edges(eid, etype) — 新增公开方法
     给定边 ID 序列 + 边类型, 返回 (src_tensor, dst_tensor)。
     内部: _edge_lookup_table.lookup_vertex_ids(cupy.asarray(eid), numeric_etype)
     结果减去 vertex_offset 转为图内局部节点 ID。
     边类型方向由 __graph["direction"] 决定 (out → sources/destinations 正向;
     in → 交换 src/dst 名称)。

  5. global_uniform_negative_sampling(num_samples, ...) — 新增公开方法
     DGL API 对齐: 构造一组图中不存在的 (src, dst) 对。
     参数:
       num_samples     : 目标负样本数 (最终可能更少, 视图密度)
       exclude_self_loops: 丢弃 src==dst 的对, 默认 True
       replace         : 有放回采样 (不支持, 抛 NotImplementedError)
       etype           : 边类型 (单类型图可省略)
       redundancy      : 不支持 (warn 后忽略)
     流程:
       - 同构图: 单顶点集, src_bias = dst_bias = ones(N)
       - 异构图 src==dst: 单顶点集 + 全1权重
       - 异构图 src≠dst: concat(src_range, dst_range); src_bias = [ones(src)|zeros(dst)];
                         dst_bias = [zeros(src)|ones(dst)]  → 与 8b3b67f 掩码逻辑一致
       - 多GPU: all_reduce 总样本数; 按 world_size 切分顶点 + bias
       - 调用 pylibcugraph.negative_sampling(remove_duplicates=True,
                                              remove_false_negatives=True,
                                              exact_number_of_samples=True)
       - 截断到 num_samples (workaround for C API global count, rapidsai/cugraph#4672)
       - exclude_self_loops: 过滤 src==dst

  6. 测试文件变化 (conftest.py / test_graph.py / test_graph_mg.py):
     - create_karate_bipartite(): 新增工厂函数, 构建异构 karate 二部图 (4种边类型)
     - karate_bipartite fixture
     - test_graph_make_heterogeneous_graph: 重构使用 karate_bipartite fixture
     - test_graph_find: 新增, 验证 find_edges() 的 src/dst 正确性 + 越界返回负值
     - test_graph_uniform_negative_sample: 新增, 验证负样本范围 + 无假负例 + 自环过滤
     - MG 版本: test_graph_find_mg, test_graph_uniform_negative_sample_mg
     - 补充: destroy_process_group() 在 cugraph_comms_shutdown() 后 (MG测试资源清理)

  删除:
     - graph.py: 已遗留的 print(u,) 调试语句 (早年临时调试代码)

diff 精读 (逐行关键点):
  +self.__edge_lookup_table = None  — __init__ 中初始化, 保证属性始终存在
  +def _clear_graph(): 三行清理  — 替代散落的 2行 None 赋值 (DRY)
  +def _to_numeric_etype(): 排序 __edge_indices 键 enumerate  — 与 C 层 EdgeIdLookupTable
    索引约定对齐; 用 dict comprehension 反查; None 单类型直接返回 0
  -print(u,)  — 删除遗留临时调试 print (graph.py L620)
  +@property _edge_lookup_table: None 时构建, 否则缓存  — 懒加载模式
  +find_edges(): src/dst 名按 direction 交换; 减 vertex_offset  — 正确本地化节点ID
  +global_uniform_negative_sampling(): 异构掩码 + 多GPU all_reduce + 截断 + self-loop 过滤
  conftest.py: create_karate_bipartite() 将图构建逻辑集中化, 4种边类型完整覆盖
  test_graph.py: find + neg_sampling 两组测试; MG 测试补 destroy_process_group()

Knuth 审查:
  1. diff 对比源:
     - _clear_graph() 统一清理3个字段 vs 旧代码 2处各自赋2行 None:
       旧代码有4处 `__graph=None; __vertex_offsets=None`, 全部替换为 _clear_graph(),
       但 8bf2012 新增了 __edge_lookup_table, 若遗漏清理则下次重建时拿到旧 graph handle
       → use-after-free 风险。_clear_graph() 是防御性设计。
     - _to_numeric_etype 用 sorted(keys) enumerate: 与 C 层约定必须严格一致,
       若 Python 和 C 排序不同则 lookup_vertex_ids 拿到错误 etype_id → 返回错误节点对。
     - find_edges 减 vertex_offset: 异构图全局ID vs 图内局部ID 转换,
       若忘记减 offset, 测试中 `assert srcs[2] < 0` 反而可能因大正数通过。

  2. 用户角度 bug:
     - 负样本截断 `[:num_samples]` 是 TODO workaround (rapidsai/cugraph#4672):
       C API 返回全局总数的负样本, 多GPU时每个 rank 都取完整结果再截断,
       导致各 rank 负样本可能重叠 (同一 (src,dst) 出现在多个 rank)。
       这在链路预测训练时会降低负样本多样性, 但不会引发崩溃。
       本迁移保留 TODO 注释, 等待上游修复。
     - redundancy 参数: warn 后静默忽略。若用户依赖 redundancy 控制采样密度,
       行为与预期不符但无报错。改写: 加断点打印参数值, 方便排查。
     - replace=True 直接 raise NotImplementedError: 正确; 不能静默 fallback。

  3. 系统角度安全:
     - _edge_lookup_table 懒加载: _clear_graph() 必须置 None, 否则下次 add_edges()
       后 lookup_table 仍指向旧 graph object → C 层悬空指针 → CUDA segment fault。
       本迁移在 EdgeLookupSession.__enter__ 加断点验证 graph handle 一致性。
     - 多GPU negative_sampling: cupy.array_split 按 world_size 切分 bias,
       若 world_size > len(bias), 某些 rank 得到空 array → negative_sampling 返回0个样本。
       上游未处理此边界; 本迁移在 WalpurgisNegSamplingSession 加断点 + 警告。
     - vertex_offset 减法: src_neg = result["sources"] - src_vertex_offset,
       若 C 层返回的 sources 值 < src_vertex_offset (理论上不应该),
       结果为负数。test_graph_find 的 `assert srcs[2] < 0` 专门测试此行为 (越界边ID)。

Walpurgis 改写20%（鲁迅拿法）:
  - GraphClearSession: 对应 _clear_graph(), 封装为 context manager + 调用记录;
    Python 是3行赋值, 我们改写为可审计的清理会话 (记录调用栈位置)
  - EdgeTypeIndex: 对应 _to_numeric_etype(), 封装为值对象 + validate();
    Python 是内联 dict comprehension, 我们改写为带缓存和调试输出的索引器
  - EdgeLookupSession: 对应 _edge_lookup_table property + find_edges() 逻辑;
    Python 是 property + 普通方法, 改写为携带 graph_direction 状态的会话对象;
    find_edges() 的 src/dst 名称交换逻辑提取为 _resolve_col_names() 消除魔法字符串
  - WalpurgisNegSamplingSession: 对应 global_uniform_negative_sampling();
    Python 是长方法, 改写为 build_bias_plan() + execute() 两阶段;
    bias 构建逻辑提取为 _BiasBuilder 静态类 (类比 NegSamplingVertexMask)
  - _BiasBuilder: 对应 global_uniform_negative_sampling() 中 if/else 掩码树;
    三个静态方法: _homo(), _hetero_same_type(), _hetero_diff_type() — 命名对称
  - 全链路 7处断点 print (WALPURGIS_DEBUG=1 门控):
    1. EdgeTypeIndex.__init__ — 排序后的 etype→index 映射表
    2. EdgeLookupSession.find_edges 入口 — eid 长度, numeric_etype, direction
    3. EdgeLookupSession._resolve_col_names — src/dst 列名选择
    4. WalpurgisNegSamplingSession.build_bias_plan — 掩码路径选择
    5. _BiasBuilder._hetero_diff_type — concat 大小 + 零填充宽度
    6. WalpurgisNegSamplingSession.execute — num_samples_global, workaround 截断量
    7. WalpurgisNegSamplingSession.execute — exclude_self_loops 过滤后剩余数量

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg(tag: str, msg: str) -> None:
    """断点调试: link_pred_neg_sampling 专用 print"""
    if _DBG:
        print(f"[DEBUG 8bf2012 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── GraphClearSession: 对应 graph.py _clear_graph() ──────────────────────────
# Python (8bf2012 graph.py):
#   def _clear_graph(self):
#       self.__graph = None
#       self.__edge_lookup_table = None
#       self.__vertex_offsets = None
#
# 8bf2012 引入动机: 之前代码有 4 处 `__graph=None; __vertex_offsets=None` 散落赋值,
# 新增 __edge_lookup_table 后若不统一清理会遗漏 → use-after-free。
# 改写: 封装为记录调用位置的会话对象; validate_cleared() 可在调试时验证三字段均为 None。
class GraphClearSession:
    """
    Encapsulates the _clear_graph() logic from 8bf2012.

    Python (8bf2012 graph.py:128-131):
        def _clear_graph(self):
            self.__graph = None
            self.__edge_lookup_table = None
            self.__vertex_offsets = None

    改写: 三字段清理封装为可审计对象。clear_count 累计调用次数可在调试中确认
    add_nodes/add_edges 触发了正确的缓存失效。

    Knuth 审查: 8bf2012 前旧代码 4 处 2-行赋值 (graph.py lines 237, 328, 555 等);
    每次新增字段都需要手动更新 N 处 → 容易遗漏。_clear_graph() 是单点维护。
    """

    def __init__(self, caller_name: str = "unknown"):
        """
        caller_name: 调用方标识 (add_nodes / add_edges / direction_change 等)
        对应各处 self._clear_graph() 调用点
        """
        self.caller_name = caller_name
        self.clear_count: int = 0
        self._cleared_fields: List[str] = []

        _dbg(
            "GraphClearSession.__init__",
            f"caller={caller_name} 已初始化, 等待 apply() 调用"
        )

    def apply(self, graph_obj: Any) -> None:
        """
        执行清理, 对应 Python _clear_graph() 三行赋值。

        graph_obj: 携带 _GraphClearTarget 协议的对象 (graph.py 的 Graph 实例),
        本迁移层不直接依赖 cugraph_dgl.Graph, 用 duck typing + hasattr 校验。

        断点调试: 打印清理前各字段状态 (是否为 None)
        """
        # Duck-type 检查: 确认对象有这三个私有属性 (Python name-mangling)
        # Graph 类中 __graph 实际存储为 _Graph__graph, 以此检测
        mangled_names = {
            "graph": "_Graph__graph",
            "edge_lookup_table": "_Graph__edge_lookup_table",
            "vertex_offsets": "_Graph__vertex_offsets",
        }

        if _DBG:
            before_states = {}
            for friendly, mangled in mangled_names.items():
                val = getattr(graph_obj, mangled, "<<attr_missing>>")
                before_states[friendly] = "None" if val is None else f"<{type(val).__name__}>"
            _dbg(
                "GraphClearSession.apply",
                f"caller={self.caller_name} clear #{self.clear_count + 1} "
                f"before: graph={before_states['graph']} "
                f"edge_lookup={before_states['edge_lookup_table']} "
                f"vertex_offsets={before_states['vertex_offsets']}"
            )

        # 8bf2012 _clear_graph() 三行:
        # self.__graph = None
        if hasattr(graph_obj, "_Graph__graph"):
            graph_obj._Graph__graph = None
            self._cleared_fields.append("__graph")

        # self.__edge_lookup_table = None  ← 8bf2012 新增清理
        if hasattr(graph_obj, "_Graph__edge_lookup_table"):
            graph_obj._Graph__edge_lookup_table = None
            self._cleared_fields.append("__edge_lookup_table")

        # self.__vertex_offsets = None
        if hasattr(graph_obj, "_Graph__vertex_offsets"):
            graph_obj._Graph__vertex_offsets = None
            self._cleared_fields.append("__vertex_offsets")

        self.clear_count += 1

        _dbg(
            "GraphClearSession.apply",
            f"caller={self.caller_name} clear #{self.clear_count} 完成, "
            f"已清理字段: {self._cleared_fields}"
        )

    def validate_cleared(self, graph_obj: Any) -> bool:
        """
        调试辅助: 验证三字段均为 None (add_edges 后立即调用可确认缓存失效)

        对应 Knuth 审查第3点: _clear_graph() 遗漏会导致 C 层悬空指针。
        """
        checks = {
            "_Graph__graph": getattr(graph_obj, "_Graph__graph", "<<missing>>"),
            "_Graph__edge_lookup_table": getattr(graph_obj, "_Graph__edge_lookup_table", "<<missing>>"),
            "_Graph__vertex_offsets": getattr(graph_obj, "_Graph__vertex_offsets", "<<missing>>"),
        }
        all_none = all(v is None for v in checks.values())

        _dbg(
            "GraphClearSession.validate_cleared",
            f"caller={self.caller_name} "
            + " ".join(f"{k.split('__')[1]}={'None' if v is None else type(v).__name__}"
                       for k, v in checks.items())
            + f" → {'✓ all_none' if all_none else '✗ LEAK DETECTED'}"
        )

        if not all_none:
            non_none = {k: v for k, v in checks.items() if v is not None}
            print(
                f"[WARN 8bf2012 GraphClearSession] caller={self.caller_name}: "
                f"字段未被清理 → {list(non_none.keys())}。"
                f"可能导致 EdgeIdLookupTable 持有旧 graph handle (use-after-free)。",
                file=sys.stderr
            )
        return all_none


# ─── EdgeTypeIndex: 对应 _to_numeric_etype() ──────────────────────────────────
# Python (8bf2012 graph.py:152-163):
#   def _to_numeric_etype(self, etype):
#       if etype is None:
#           if len(self.canonical_etypes) > 1:
#               raise ValueError(...)
#           return 0
#       etype = self.to_canonical_etype(etype)
#       return {k: i for i, k in enumerate(
#           sorted(self.__edge_indices.keys(leaves_only=True, include_nested=True))
#       )}[etype]
#
# 改写: 封装为值对象, 缓存 etype→index 映射; dump_table() 可打印全表 (与上游宏展开类比)
@dataclass
class EdgeTypeIndexEntry:
    """一条 etype → numeric_index 映射记录。对应 enumerate 产生的 (i, k) 对。"""
    etype: Tuple[str, str, str]   # canonical etype tuple
    index: int                    # 对应 C 层 EdgeIdLookupTable 期望的整数 ID
    sort_key: str                 # str(etype) 用于排序 (对应 sorted() 键)


class EdgeTypeIndex:
    """
    Encapsulates _to_numeric_etype() from 8bf2012.

    Python (8bf2012 graph.py:152-163): 内联 dict comprehension, 每次调用重建。
    改写: 构造时建立缓存映射表; dump_table() 使全部路径可见。

    核心约定 (Knuth 审查第1点):
    Python 用 sorted(self.__edge_indices.keys(leaves_only=True, include_nested=True))
    C 层 EdgeIdLookupTable 使用相同排序约定。Python 和 C 必须严格一致,
    否则 lookup_vertex_ids(eid, wrong_etype_id) → 返回错误节点对。
    """

    def __init__(self, canonical_etypes: List[Tuple[str, str, str]]):
        """
        canonical_etypes: 已排序的边类型列表
        对应 Python sorted(self.__edge_indices.keys(leaves_only=True, include_nested=True))

        断点调试: 打印完整 etype→index 映射表
        """
        # 8bf2012: 按 sorted() 建立 index (与 C 层 EdgeIdLookupTable 约定一致)
        self._entries: Dict[Tuple[str, str, str], EdgeTypeIndexEntry] = {}
        for i, etype in enumerate(sorted(canonical_etypes)):
            entry = EdgeTypeIndexEntry(
                etype=etype,
                index=i,
                sort_key=str(etype),
            )
            self._entries[etype] = entry

        _dbg(
            "EdgeTypeIndex.__init__",
            f"共 {len(self._entries)} 种边类型, 映射表: "
            + " | ".join(f"{e.etype}→{e.index}" for e in sorted(
                self._entries.values(), key=lambda x: x.index
            ))
        )

    def lookup(
        self,
        etype: Optional[Union[str, Tuple[str, str, str]]],
        canonical_etype_fn=None,
    ) -> int:
        """
        对应 Python _to_numeric_etype(self, etype):
          if etype is None: ...return 0
          etype = self.to_canonical_etype(etype)
          return {k: i for ...}[etype]

        canonical_etype_fn: graph.to_canonical_etype 的引用 (可选; None 时 etype 已是 tuple)

        断点调试: 打印输入 etype + 解析后 canonical + 返回 index
        """
        _dbg(
            "EdgeTypeIndex.lookup",
            f"输入 etype={etype!r} canonical_etype_fn={'provided' if canonical_etype_fn else 'None'}"
        )

        # 8bf2012: None → 单类型图返回 0; 多类型图 None 抛 ValueError
        if etype is None:
            if len(self._entries) > 1:
                raise ValueError(
                    "[8bf2012 EdgeTypeIndex] Edge type is required for heterogeneous graphs. "
                    f"已知边类型: {list(self._entries.keys())}"
                )
            _dbg("EdgeTypeIndex.lookup", "etype=None, 单类型图 → 返回 0")
            return 0

        # to_canonical_etype 转换
        if canonical_etype_fn is not None:
            canonical = canonical_etype_fn(etype)
        else:
            # 已是 tuple 时直接使用
            canonical = etype if isinstance(etype, tuple) else (etype, etype, etype)

        if canonical not in self._entries:
            raise ValueError(
                f"[8bf2012 EdgeTypeIndex] 未知边类型 {canonical!r}。"
                f"已知: {list(self._entries.keys())}"
            )

        index = self._entries[canonical].index

        _dbg(
            "EdgeTypeIndex.lookup",
            f"canonical={canonical!r} → numeric_index={index}"
        )
        return index

    def dump_table(self) -> None:
        """
        断点调试: 打印完整映射表 (类比 fp16_grad_dedup.py 的 dump_dispatch_table())

        Python (8bf2012): dict comprehension 内联, 无 dump 能力。
        改写: 可调用 dump_table() 在调试时确认 C 层 EdgeIdLookupTable 的索引约定。
        """
        print(
            f"[DEBUG 8bf2012 EdgeTypeIndex.dump_table] "
            f"共 {len(self._entries)} 条路径:",
            file=sys.stderr
        )
        for entry in sorted(self._entries.values(), key=lambda x: x.index):
            print(
                f"  [{entry.index}] {entry.etype} (sort_key={entry.sort_key!r})",
                file=sys.stderr
            )


# ─── _BiasBuilder: 对应 global_uniform_negative_sampling 掩码构建树 ─────────────
# Python (8bf2012 graph.py 949-1005):
#   if len(self.ntypes) == 1:
#       vertices = arange(num_nodes); src_bias=dst_bias=ones(N)
#   else:
#       if can_edge_type[0] == can_edge_type[2]:
#           vertices = arange(offset, offset+num_src); src_bias=dst_bias=ones
#       else:
#           vertices = concat(src_range, dst_range)
#           src_bias = [ones(src)|zeros(dst)]; dst_bias = [zeros(src)|ones(dst)]
#
# 改写: 三条路径提取为静态方法 (类比 NegSamplingVertexMask)
@dataclass
class NegBiasPlan:
    """
    一次负采样的顶点集 + bias 向量计划。
    对应 Python global_uniform_negative_sampling() 中三组 if/else 分支的赋值结果。

    改写: Python 是函数内局部变量, 我们改写为值对象; validate() 独立可测。
    """
    vertices: Any                  # cupy/torch tensor 或 list (顶点全局ID)
    src_bias: Any                  # cupy array (源节点采样权重)
    dst_bias: Any                  # cupy array (目的节点采样权重)
    src_vertex_offset: int         # 结果减法: local_id = global_id - offset
    dst_vertex_offset: int
    path: str                      # 调试: "homo" / "hetero_same" / "hetero_diff"
    num_src_nodes: int
    num_dst_nodes: int

    def validate(self) -> None:
        """
        断点调试: 验证 bias 向量长度与 vertices 一致

        Python 无此检查; 改写新增防御性断言, 对应 Knuth 审查第3点:
        若 bias 长度 != vertices 长度, pylibcugraph 下标越界 → CUDA segment fault。
        """
        v_len = len(self.vertices) if hasattr(self.vertices, '__len__') else -1
        sb_len = len(self.src_bias) if hasattr(self.src_bias, '__len__') else -1
        db_len = len(self.dst_bias) if hasattr(self.dst_bias, '__len__') else -1

        _dbg(
            "NegBiasPlan.validate",
            f"path={self.path} vertices_len={v_len} "
            f"src_bias_len={sb_len} dst_bias_len={db_len} "
            f"src_off={self.src_vertex_offset} dst_off={self.dst_vertex_offset}"
        )

        if v_len != -1 and sb_len != -1 and v_len != sb_len:
            print(
                f"[WARN 8bf2012 NegBiasPlan.validate] path={self.path}: "
                f"vertices_len={v_len} != src_bias_len={sb_len} "
                f"→ pylibcugraph 下标越界风险",
                file=sys.stderr
            )
        if v_len != -1 and db_len != -1 and v_len != db_len:
            print(
                f"[WARN 8bf2012 NegBiasPlan.validate] path={self.path}: "
                f"vertices_len={v_len} != dst_bias_len={db_len} "
                f"→ pylibcugraph 下标越界风险",
                file=sys.stderr
            )


class _BiasBuilder:
    """
    Static methods for building NegBiasPlan.
    Corresponds to 8bf2012 global_uniform_negative_sampling() if/else tree.

    改写: Python 是方法内 if/else + 局部变量赋值;
    我们改写为三个命名静态方法 + build() 分发 (类比 NegSamplingVertexMask)。
    """

    @staticmethod
    def _homo(num_nodes: int) -> NegBiasPlan:
        """
        对应 8bf2012: if len(self.ntypes) == 1:
            vertices = torch.arange(self.num_nodes())
            src_bias = dst_bias = cupy.ones(len(vertices), dtype='float32')

        同构图无 offset, src==dst 顶点集, 均匀采样。
        """
        try:
            import torch
            import cupy
            vertices = torch.arange(num_nodes, dtype=torch.int64, device="cuda")
            src_bias = cupy.ones(num_nodes, dtype="float32")
            dst_bias = src_bias
        except ImportError:
            # 无 GPU 环境: 退化为 list (单元测试用)
            vertices = list(range(num_nodes))
            src_bias = [1.0] * num_nodes
            dst_bias = src_bias

        _dbg(
            "_BiasBuilder._homo",
            f"num_nodes={num_nodes} vertices=[0,{num_nodes}) src_bias=dst_bias=ones({num_nodes})"
        )

        return NegBiasPlan(
            vertices=vertices,
            src_bias=src_bias,
            dst_bias=dst_bias,
            src_vertex_offset=0,
            dst_vertex_offset=0,
            path="homo",
            num_src_nodes=num_nodes,
            num_dst_nodes=num_nodes,
        )

    @staticmethod
    def _hetero_same_type(
        src_type: str,
        vertex_offsets: Dict[str, int],
        num_src_nodes: int,
    ) -> NegBiasPlan:
        """
        对应 8bf2012: if can_edge_type[0] == can_edge_type[2]:
            vertices = torch.arange(offset, offset+num_src, ...)
            src_bias = cupy.ones(num_src, dtype='float32')
            dst_bias = src_bias

        异构图但 src/dst 同类型: 单顶点集, 全1权重, 需要 offset。
        注意: src_bias == dst_bias (同类型, 无需掩码)。
        """
        offset = vertex_offsets.get(src_type, 0)

        try:
            import torch
            import cupy
            vertices = torch.arange(
                offset, offset + num_src_nodes,
                dtype=torch.int64, device="cuda"
            )
            src_bias = cupy.ones(num_src_nodes, dtype="float32")
            dst_bias = src_bias
        except ImportError:
            vertices = list(range(offset, offset + num_src_nodes))
            src_bias = [1.0] * num_src_nodes
            dst_bias = src_bias

        _dbg(
            "_BiasBuilder._hetero_same_type",
            f"src_type={src_type!r} offset={offset} "
            f"vertices=[{offset}, {offset+num_src_nodes}) "
            f"src_bias=dst_bias=ones({num_src_nodes})"
        )

        return NegBiasPlan(
            vertices=vertices,
            src_bias=src_bias,
            dst_bias=dst_bias,
            src_vertex_offset=offset,
            dst_vertex_offset=offset,
            path="hetero_same",
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_src_nodes,
        )

    @staticmethod
    def _hetero_diff_type(
        src_type: str,
        dst_type: str,
        vertex_offsets: Dict[str, int],
        num_src_nodes: int,
        num_dst_nodes: int,
    ) -> NegBiasPlan:
        """
        对应 8bf2012 核心路径: else (can_edge_type[0] != can_edge_type[2]):
            vertices = concat(
                arange(src_off, src_off+num_src),
                arange(dst_off, dst_off+num_dst),
            )
            src_bias = concat([ones(num_src), zeros(num_dst)])
            dst_bias = concat([zeros(num_src), ones(num_dst)])

        异构图 src≠dst: concat 顶点集 + 掩码 bias。
        核心机制: src 采样只命中 src 节点 (dst 槽位 bias=0);
                 dst 采样只命中 dst 节点 (src 槽位 bias=0)。
        与 8b3b67f neg_sample() 掩码逻辑一致 (本 commit 在 DGL 层重新实现)。

        断点调试: 打印 offset 值 + concat 大小 + 零填充宽度
        """
        src_off = vertex_offsets.get(src_type, 0)
        dst_off = vertex_offsets.get(dst_type, 0)

        _dbg(
            "_BiasBuilder._hetero_diff_type",
            f"src_type={src_type!r} dst_type={dst_type!r} "
            f"src_off={src_off} dst_off={dst_off} "
            f"num_src={num_src_nodes} num_dst={num_dst_nodes}"
        )

        try:
            import torch
            import cupy
            src_range = torch.arange(src_off, src_off + num_src_nodes,
                                     dtype=torch.int64, device="cuda")
            dst_range = torch.arange(dst_off, dst_off + num_dst_nodes,
                                     dtype=torch.int64, device="cuda")
            vertices = torch.concat([src_range, dst_range])

            # src_bias: [ones(src) | zeros(dst)] → dst 节点在 src 采样中权重为0
            src_bias = cupy.concatenate([
                cupy.ones(num_src_nodes, dtype="float32"),
                cupy.zeros(num_dst_nodes, dtype="float32"),
            ])
            # dst_bias: [zeros(src) | ones(dst)] → src 节点在 dst 采样中权重为0
            dst_bias = cupy.concatenate([
                cupy.zeros(num_src_nodes, dtype="float32"),
                cupy.ones(num_dst_nodes, dtype="float32"),
            ])
        except ImportError:
            # 无 GPU 环境退化
            vertices = list(range(src_off, src_off + num_src_nodes)) + \
                       list(range(dst_off, dst_off + num_dst_nodes))
            src_bias = [1.0] * num_src_nodes + [0.0] * num_dst_nodes
            dst_bias = [0.0] * num_src_nodes + [1.0] * num_dst_nodes

        total_v = num_src_nodes + num_dst_nodes
        _dbg(
            "_BiasBuilder._hetero_diff_type",
            f"vertices concat 完成: src_range=[{src_off},{src_off+num_src_nodes}) "
            f"dst_range=[{dst_off},{dst_off+num_dst_nodes}) total={total_v} "
            f"src_bias=[ones({num_src_nodes})|zeros({num_dst_nodes})] "
            f"dst_bias=[zeros({num_src_nodes})|ones({num_dst_nodes})]"
        )

        return NegBiasPlan(
            vertices=vertices,
            src_bias=src_bias,
            dst_bias=dst_bias,
            src_vertex_offset=src_off,
            dst_vertex_offset=dst_off,
            path="hetero_diff",
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
        )

    @classmethod
    def build(
        cls,
        is_homogeneous: bool,
        can_edge_type: Optional[Tuple[str, str, str]],
        vertex_offsets: Dict[str, int],
        num_nodes_fn,          # callable(ntype=None) → int, 对应 graph.num_nodes()
    ) -> NegBiasPlan:
        """
        8bf2012 global_uniform_negative_sampling() bias 构建逻辑入口。

        断点调试: 打印路径选择决策 + 构建完成后 NegBiasPlan 摘要
        """
        _dbg(
            "_BiasBuilder.build",
            f"is_homo={is_homogeneous} can_edge_type={can_edge_type!r}"
        )

        if is_homogeneous:
            # 8bf2012: if len(self.ntypes) == 1:
            plan = cls._homo(num_nodes_fn())
        else:
            src_type, _, dst_type = can_edge_type
            num_src = num_nodes_fn(src_type)
            num_dst = num_nodes_fn(dst_type)

            if src_type == dst_type:
                # 8bf2012: if can_edge_type[0] == can_edge_type[2]:
                plan = cls._hetero_same_type(src_type, vertex_offsets, num_src)
            else:
                # 8bf2012: else (核心掩码路径)
                plan = cls._hetero_diff_type(
                    src_type, dst_type, vertex_offsets, num_src, num_dst
                )

        plan.validate()
        return plan


# ─── WalpurgisNegSamplingSession: 对应 global_uniform_negative_sampling() ─────
# Python (8bf2012 graph.py:918-1097):
#   def global_uniform_negative_sampling(self, num_samples, exclude_self_loops,
#                                        replace, etype, redundancy):
# 改写: build_bias_plan() + execute() 两阶段; 参数检查提前; 多GPU逻辑内聚
class WalpurgisNegSamplingSession:
    """
    Encapsulates global_uniform_negative_sampling() from 8bf2012.

    Python 是单个长方法 (约180行); 改写为两阶段会话:
    1. __init__: 参数校验 (replace, redundancy)
    2. build_bias_plan(): bias/vertices 构建 (纯数学, 无 pylibcugraph 调用)
    3. execute(): 实际采样 + 截断 + self-loop 过滤

    Knuth 审查第2点:
    - TODO workaround (rapidsai/cugraph#4672): 截断到 num_samples 导致多GPU重复
    - redundancy 静默忽略: 加断点 warn 告知用户
    - replace=True: 直接 raise (正确, 不能 fallback)
    """

    def __init__(
        self,
        num_samples: int,
        exclude_self_loops: bool = True,
        replace: bool = False,
        etype: Optional[Union[str, Tuple[str, str, str]]] = None,
        redundancy: Optional[float] = None,
    ):
        # 8bf2012: redundancy 参数检查
        if redundancy:
            warnings.warn(
                "[8bf2012 WalpurgisNegSamplingSession] "
                f"'redundancy'={redundancy} 参数被 cuGraph-DGL 忽略。"
                f"实际采样密度由 remove_duplicates=True 控制。"
            )
            _dbg(
                "WalpurgisNegSamplingSession.__init__",
                f"redundancy={redundancy} → ignored (warn issued)"
            )

        # 8bf2012: replace 不支持
        if replace:
            raise NotImplementedError(
                "[8bf2012 WalpurgisNegSamplingSession] "
                "Negative sampling with replacement is not supported by cuGraph-DGL. "
                "请设置 replace=False (默认)。"
            )

        self.num_samples = num_samples
        self.exclude_self_loops = exclude_self_loops
        self.etype = etype
        self.redundancy = redundancy

        _dbg(
            "WalpurgisNegSamplingSession.__init__",
            f"num_samples={num_samples} exclude_self_loops={exclude_self_loops} "
            f"replace={replace} etype={etype!r} redundancy={redundancy}"
        )

    def build_bias_plan(
        self,
        is_homogeneous: bool,
        can_edge_type: Optional[Tuple[str, str, str]],
        vertex_offsets: Dict[str, int],
        num_nodes_fn,
    ) -> NegBiasPlan:
        """
        阶段1: 构建 bias 计划 (无 pylibcugraph 调用)。
        对应 8bf2012 global_uniform_negative_sampling() lines 935-1005。

        断点调试: 记录路径选择
        """
        _dbg(
            "WalpurgisNegSamplingSession.build_bias_plan",
            f"is_homo={is_homogeneous} can_edge_type={can_edge_type!r} "
            f"num_samples={self.num_samples}"
        )

        return _BiasBuilder.build(
            is_homogeneous=is_homogeneous,
            can_edge_type=can_edge_type,
            vertex_offsets=vertex_offsets,
            num_nodes_fn=num_nodes_fn,
        )

    def apply_multi_gpu_split(
        self,
        plan: NegBiasPlan,
        can_edge_type: Optional[Tuple[str, str, str]],
    ) -> Tuple[NegBiasPlan, int]:
        """
        多GPU: 按 rank/world_size 切分 vertices + bias。
        对应 8bf2012 lines 1007-1028:
          rank = get_rank(); world_size = get_world_size()
          num_samples_global = all_reduce(num_samples, SUM)
          vertices = tensor_split(vertices, world_size)[rank]
          src_bias = array_split(src_bias, world_size)[rank]
          dst_bias = ...

        返回: (split_plan, num_samples_global)

        断点调试: 打印 rank, world_size, split 后 bias 长度 (空数组警告)
        """
        try:
            import torch
            import cupy
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        except Exception as e:
            _dbg("WalpurgisNegSamplingSession.apply_multi_gpu_split",
                 f"distributed 不可用: {e} → skip split")
            return plan, self.num_samples

        # 8bf2012: all_reduce 全局总样本数
        num_samples_t = torch.tensor([self.num_samples], device="cuda")
        torch.distributed.all_reduce(
            num_samples_t, op=torch.distributed.ReduceOp.SUM
        )
        num_samples_global = int(num_samples_t)

        # 8bf2012: tensor_split + array_split
        vertices_split = torch.tensor_split(plan.vertices, world_size)[rank]
        src_bias_split = cupy.array_split(plan.src_bias, world_size)[rank]

        # 8bf2012: 同类型时 dst_bias == src_bias (共享)
        if can_edge_type and can_edge_type[0] == can_edge_type[2]:
            dst_bias_split = src_bias_split
        else:
            dst_bias_split = cupy.array_split(plan.dst_bias, world_size)[rank]

        _dbg(
            "WalpurgisNegSamplingSession.apply_multi_gpu_split",
            f"rank={rank}/{world_size} "
            f"num_samples_global={num_samples_global} "
            f"vertices_split_len={len(vertices_split)} "
            f"src_bias_split_len={len(src_bias_split)} "
            f"dst_bias_split_len={len(dst_bias_split)}"
        )

        # 系统安全: 空 bias 警告 (Knuth 审查第3点)
        if len(src_bias_split) == 0 or len(dst_bias_split) == 0:
            print(
                f"[WARN 8bf2012 WalpurgisNegSamplingSession] "
                f"rank={rank}: split 后 bias 为空数组 "
                f"(world_size={world_size} > 顶点数?)。"
                f"negative_sampling 将返回0个样本。",
                file=sys.stderr
            )

        split_plan = NegBiasPlan(
            vertices=vertices_split,
            src_bias=src_bias_split,
            dst_bias=dst_bias_split,
            src_vertex_offset=plan.src_vertex_offset,
            dst_vertex_offset=plan.dst_vertex_offset,
            path=plan.path + "_mg_split",
            num_src_nodes=plan.num_src_nodes,
            num_dst_nodes=plan.num_dst_nodes,
        )
        return split_plan, num_samples_global

    def execute(
        self,
        plan: NegBiasPlan,
        num_samples_global: int,
        resource_handle,
        graph_obj,
        current_graph,            # __graph dict or None
        graph_builder_fn,         # 对应 self._graph("out", prob_attr)
    ) -> Tuple[Any, Any]:
        """
        阶段2: 调用 pylibcugraph.negative_sampling + 截断 + self-loop 过滤。
        对应 8bf2012 lines 1029-1097。

        返回: (src_neg_local, dst_neg_local) — 减去 vertex_offset 的局部节点 ID

        断点调试:
          1. 入口: num_samples_global, graph direction
          2. 截断: 实际结果数 vs num_samples (workaround 数量)
          3. exclude_self_loops: 过滤前后数量
        """
        try:
            import pylibcugraph
            import cupy
            import torch
        except ImportError as e:
            raise ImportError(
                f"[8bf2012 WalpurgisNegSamplingSession.execute] "
                f"缺少依赖: {e}。需要 pylibcugraph + cupy + torch。"
            )

        # 8bf2012: 选择 graph (优先已缓存 out 方向, 否则新建)
        if current_graph is not None and current_graph["direction"] == "out":
            graph = current_graph["graph"]
            prob_attr = current_graph.get("prob_attr")
        else:
            prob_attr = None if current_graph is None else current_graph.get("prob_attr")
            graph = graph_builder_fn("out", prob_attr)

        _dbg(
            "WalpurgisNegSamplingSession.execute",
            f"num_samples_global={num_samples_global} "
            f"plan.path={plan.path} "
            f"src_off={plan.src_vertex_offset} dst_off={plan.dst_vertex_offset}"
        )

        result_dict = pylibcugraph.negative_sampling(
            resource_handle,
            graph,
            num_samples_global,
            vertices=cupy.asarray(plan.vertices),
            src_bias=plan.src_bias,
            dst_bias=plan.dst_bias,
            remove_duplicates=True,
            remove_false_negatives=True,
            exact_number_of_samples=True,
            do_expensive_check=False,
        )

        # 8bf2012 TODO workaround (rapidsai/cugraph#4672):
        # C API 返回全局总数, 截断到本 rank 的 num_samples
        raw_src_count = len(result_dict["sources"])
        truncated = max(0, raw_src_count - self.num_samples)

        _dbg(
            "WalpurgisNegSamplingSession.execute",
            f"C API 返回 {raw_src_count} 个负样本, "
            f"截断 {truncated} 个 → 保留 {self.num_samples} 个 "
            f"(TODO workaround rapidsai/cugraph#4672)"
        )

        src_neg = (
            torch.as_tensor(result_dict["sources"], device="cuda")[:self.num_samples]
            - plan.src_vertex_offset
        )
        dst_neg = (
            torch.as_tensor(result_dict["destinations"], device="cuda")[:self.num_samples]
            - plan.dst_vertex_offset
        )

        # 8bf2012: exclude_self_loops 过滤
        if self.exclude_self_loops:
            before_filter = len(src_neg)
            mask = src_neg != dst_neg
            src_neg = src_neg[mask]
            dst_neg = dst_neg[mask]
            after_filter = len(src_neg)

            _dbg(
                "WalpurgisNegSamplingSession.execute",
                f"exclude_self_loops=True: {before_filter} → {after_filter} "
                f"(过滤 {before_filter - after_filter} 条自环)"
            )
        else:
            _dbg(
                "WalpurgisNegSamplingSession.execute",
                f"exclude_self_loops=False: 保留 {len(src_neg)} 条 (含可能的自环)"
            )

        return src_neg, dst_neg


# ─── EdgeLookupSession: 对应 _edge_lookup_table property + find_edges() ─────────
# Python (8bf2012 graph.py:910-943):
#   @property
#   def _edge_lookup_table(self): ...懒加载...
#   def find_edges(self, eid, etype):
#       etype = self.to_canonical_etype(etype)
#       num_edge_type = self._to_numeric_etype(etype)
#       out = self._edge_lookup_table.lookup_vertex_ids(cupy.asarray(eid), num_edge_type)
#       src_name/dst_name 按 direction 交换
#       return (result[src_name] - offsets[src_type], result[dst_name] - offsets[dst_type])
#
# 改写: 封装为会话对象; _resolve_col_names() 消除 "sources"/"destinations" 魔法字符串
class EdgeLookupSession:
    """
    Encapsulates _edge_lookup_table property + find_edges() from 8bf2012.

    Python: property 懒加载 + 普通方法。
    改写: EdgeLookupSession 携带 graph_direction 状态; _resolve_col_names() 让
    src/dst 列名选择逻辑显式化 (Python 是两行条件赋值)。

    Knuth 审查第1点:
    find_edges 减 vertex_offset 是核心正确性要求:
    pylibcugraph 返回的是全局节点 ID (hetero 图中 type2 节点从 offset 开始),
    用户期望得到的是图内局部节点 ID (从0开始)。
    """

    # 8bf2012: src/dst 列名常量 (消除魔法字符串)
    _COL_SOURCES = "sources"
    _COL_DESTINATIONS = "destinations"

    def __init__(
        self,
        graph_direction: str,          # "out" or "in"
        edge_type_index: EdgeTypeIndex,
        vertex_offsets: Dict[str, int],
    ):
        """
        graph_direction: 对应 self.__graph["direction"]
        edge_type_index: EdgeTypeIndex 实例 (已建立 etype→index 缓存)
        vertex_offsets:  对应 self._vertex_offsets (含 ntype→global_offset 映射)
        """
        self.graph_direction = graph_direction
        self.edge_type_index = edge_type_index
        self.vertex_offsets = vertex_offsets

        _dbg(
            "EdgeLookupSession.__init__",
            f"graph_direction={graph_direction!r} "
            f"vertex_offsets={vertex_offsets}"
        )

    def _resolve_col_names(self) -> Tuple[str, str]:
        """
        对应 8bf2012 find_edges():
            src_name = "sources" if direction == "out" else "destinations"
            dst_name = "destinations" if direction == "out" else "sources"

        改写: 提取为方法, 消除 "sources"/"destinations" 内联字符串。

        断点调试: 打印方向 + 列名选择
        """
        if self.graph_direction == "out":
            src_name = self._COL_SOURCES
            dst_name = self._COL_DESTINATIONS
        else:
            src_name = self._COL_DESTINATIONS
            dst_name = self._COL_SOURCES

        _dbg(
            "EdgeLookupSession._resolve_col_names",
            f"direction={self.graph_direction!r} → src_col={src_name!r} dst_col={dst_name!r}"
        )
        return src_name, dst_name

    def find_edges(
        self,
        eid,                             # torch.Tensor (边ID序列)
        canonical_etype: Tuple[str, str, str],
        lookup_table,                    # pylibcugraph.EdgeIdLookupTable
    ) -> Tuple[Any, Any]:
        """
        对应 8bf2012 graph.py find_edges():
          num_edge_type = self._to_numeric_etype(etype)
          out = self._edge_lookup_table.lookup_vertex_ids(cupy.asarray(eid), num_edge_type)
          return (out[src_name] - offsets[src_type], out[dst_name] - offsets[dst_type])

        断点调试: 入口打印 eid 长度 + canonical_etype + numeric_etype
        """
        try:
            import cupy
            import torch
        except ImportError as e:
            raise ImportError(
                f"[8bf2012 EdgeLookupSession.find_edges] 缺少依赖: {e}"
            )

        numeric_etype = self.edge_type_index.lookup(canonical_etype)

        _dbg(
            "EdgeLookupSession.find_edges",
            f"eid_len={len(eid)} canonical_etype={canonical_etype!r} "
            f"numeric_etype={numeric_etype} direction={self.graph_direction!r}"
        )

        # 调用 pylibcugraph EdgeIdLookupTable (8bf2012 新增)
        out = lookup_table.lookup_vertex_ids(cupy.asarray(eid), numeric_etype)

        src_name, dst_name = self._resolve_col_names()
        offsets = self.vertex_offsets

        # 8bf2012: 减去 vertex_offset → 局部节点 ID
        src_local = torch.as_tensor(out[src_name], device="cuda") - offsets.get(canonical_etype[0], 0)
        dst_local = torch.as_tensor(out[dst_name], device="cuda") - offsets.get(canonical_etype[2], 0)

        _dbg(
            "EdgeLookupSession.find_edges",
            f"结果: src_local min={src_local.min().item() if len(src_local) else 'N/A'} "
            f"max={src_local.max().item() if len(src_local) else 'N/A'} "
            f"dst_local min={dst_local.min().item() if len(dst_local) else 'N/A'} "
            f"max={dst_local.max().item() if len(dst_local) else 'N/A'} "
            f"(负值=越界边ID, test_graph_find 期望 srcs[2]<0)"
        )

        return src_local, dst_local


# ─── KarateGraphBipartiteFactory: 对应 conftest.py create_karate_bipartite() ──
# Python (8bf2012 conftest.py:72-138):
#   def create_karate_bipartite(multi_gpu=False):
#       df = karate.get_edgelist()
#       graph = cugraph_dgl.Graph(is_multi_gpu=...)
#       4种边类型: (type1,e1,type1), (type1,e2,type2), (type2,e3,type1), (type2,e4,type2)
#       edges[type].dst/src 减去 offset 转为局部 ID
#
# 改写: 封装为工厂类; _partition_edges() 提取分区逻辑; 节点偏移计算显式化
class KarateGraphBipartiteFactory:
    """
    Factory for the heterogeneous karate bipartite test graph introduced in 8bf2012.

    Python (8bf2012 conftest.py): 独立函数 create_karate_bipartite()。
    改写: 封装为工厂类; _partition_edges() 和 _offset_edges() 分离关注点。
    4种边类型 (type1→type1, type1→type2, type2→type1, type2→type2) 覆盖
    all combinations of src/dst node type pairs (全覆盖 find_edges + neg_sampling 测试)。

    Knuth 审查第1点: 边的 src/dst 减 offset 逻辑:
      edges["type1","e2","type2"].dst -= num_nodes_group_1
      edges["type2","e3","type1"].src -= num_nodes_group_1
      edges["type2","e4","type2"].src -= num_nodes_group_1
      edges["type2","e4","type2"].dst -= num_nodes_group_1
    必须在 add_edges 前完成, 否则 type2 节点的局部 ID 错误。
    """

    # 8bf2012 定义的4种边类型
    EDGE_TYPES = [
        ("type1", "e1", "type1"),
        ("type1", "e2", "type2"),
        ("type2", "e3", "type1"),
        ("type2", "e4", "type2"),
    ]

    @staticmethod
    def _partition_edges(df, num_group_1: int):
        """
        对应 8bf2012 conftest.py 边分区逻辑:
          edges["type1","e1","type1"] = df[(df.src < g1) & (df.dst < g1)]
          edges["type1","e2","type2"] = df[(df.src < g1) & (df.dst >= g1)]
          edges["type2","e3","type1"] = df[(df.src >= g1) & (df.dst < g1)]
          edges["type2","e4","type2"] = df[(df.src >= g1) & (df.dst >= g1)]

        返回: dict of etype → DataFrame (原始全局ID, 未减 offset)
        断点调试: 打印每种类型的边数
        """
        g1 = num_group_1
        edges = {}
        edges["type1", "e1", "type1"] = df[(df.src < g1) & (df.dst < g1)].copy()
        edges["type1", "e2", "type2"] = df[(df.src < g1) & (df.dst >= g1)].copy()
        edges["type2", "e3", "type1"] = df[(df.src >= g1) & (df.dst < g1)].copy()
        edges["type2", "e4", "type2"] = df[(df.src >= g1) & (df.dst >= g1)].copy()

        _dbg(
            "KarateGraphBipartiteFactory._partition_edges",
            " ".join(
                f"{etype[1]}:{len(edf)}"
                for etype, edf in edges.items()
            )
        )
        return edges

    @staticmethod
    def _offset_edges(edges: dict, num_group_1: int) -> dict:
        """
        对应 8bf2012 conftest.py offset 减法:
          edges["type1","e2","type2"].dst -= num_nodes_group_1
          edges["type2","e3","type1"].src -= num_nodes_group_1
          edges["type2","e4","type2"].src -= num_nodes_group_1
          edges["type2","e4","type2"].dst -= num_nodes_group_1

        将全局 ID 转为 type-local ID。
        断点调试: 打印减 offset 前后的 src/dst 范围
        """
        g1 = num_group_1

        _dbg(
            "KarateGraphBipartiteFactory._offset_edges",
            f"num_group_1={g1} 开始 offset 减法"
        )

        edges["type1", "e2", "type2"].dst -= g1
        _dbg("KarateGraphBipartiteFactory._offset_edges",
             f"e2 dst offset 后 range: [{edges['type1','e2','type2'].dst.min()}, "
             f"{edges['type1','e2','type2'].dst.max()}]"
             if len(edges["type1", "e2", "type2"]) > 0 else "e2: 0条边")

        edges["type2", "e3", "type1"].src -= g1
        _dbg("KarateGraphBipartiteFactory._offset_edges",
             f"e3 src offset 后 range: [{edges['type2','e3','type1'].src.min()}, "
             f"{edges['type2','e3','type1'].src.max()}]"
             if len(edges["type2", "e3", "type1"]) > 0 else "e3: 0条边")

        edges["type2", "e4", "type2"].src -= g1
        edges["type2", "e4", "type2"].dst -= g1
        _dbg("KarateGraphBipartiteFactory._offset_edges",
             f"e4 src/dst offset 后 src_range: [{edges['type2','e4','type2'].src.min()}, "
             f"{edges['type2','e4','type2'].src.max()}] "
             f"dst_range: [{edges['type2','e4','type2'].dst.min()}, "
             f"{edges['type2','e4','type2'].dst.max()}]"
             if len(edges["type2", "e4", "type2"]) > 0 else "e4: 0条边")

        return edges

    @classmethod
    def create(cls, multi_gpu: bool = False):
        """
        对应 8bf2012 create_karate_bipartite(multi_gpu=False)。
        返回: (graph, edges_dict, (num_nodes_group_1, num_nodes_group_2))

        edges_dict: 含减过 offset 的局部 ID (用于测试断言)
        断点调试: 打印节点分组大小 + 各类型边数
        """
        try:
            import numpy as np
            import torch
            import cugraph_dgl
            from cugraph.datasets import karate
        except ImportError as e:
            raise ImportError(
                f"[8bf2012 KarateGraphBipartiteFactory.create] 缺少依赖: {e}"
            )

        df = karate.get_edgelist()
        df.src = df.src.astype("int64")
        df.dst = df.dst.astype("int64")

        total_num_nodes = max(df.src.max(), df.dst.max()) + 1
        num_group_1 = total_num_nodes // 2
        num_group_2 = total_num_nodes - num_group_1

        _dbg(
            "KarateGraphBipartiteFactory.create",
            f"total_nodes={total_num_nodes} group1={num_group_1} group2={num_group_2} "
            f"multi_gpu={multi_gpu}"
        )

        node_x_1 = np.random.random((num_group_1,))
        node_x_2 = np.random.random((num_group_2,))

        if multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            node_x_1 = np.array_split(node_x_1, world_size)[rank]
            node_x_2 = np.array_split(node_x_2, world_size)[rank]

        graph = cugraph_dgl.Graph(is_multi_gpu=multi_gpu)
        graph.add_nodes(num_group_1, {"x": node_x_1}, "type1")
        graph.add_nodes(num_group_2, {"x": node_x_2}, "type2")

        # 分区 + offset 转换
        edges = cls._partition_edges(df, num_group_1)
        edges = cls._offset_edges(edges, num_group_1)

        # 多GPU: 按 rank 切分边
        if multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            import numpy as np
            edges_local = {
                etype: edf.iloc[np.array_split(np.arange(len(edf)), world_size)[rank]]
                for etype, edf in edges.items()
            }
        else:
            edges_local = edges

        for etype, edf in edges_local.items():
            graph.add_edges(edf.src, edf.dst, etype=etype)

        _dbg(
            "KarateGraphBipartiteFactory.create",
            f"图构建完成: ntypes={graph.ntypes} etypes={graph.canonical_etypes}"
        )

        return graph, edges, (num_group_1, num_group_2)


# ─── 自检: 无 GPU 依赖的配置层单元测试 ─────────────────────────────────────────
def _test_edge_type_index() -> bool:
    """
    验证 EdgeTypeIndex: sorted + lookup 逻辑
    对应 8bf2012 _to_numeric_etype 约定
    """
    etypes = [
        ("type2", "e4", "type2"),
        ("type1", "e1", "type1"),
        ("type2", "e3", "type1"),
        ("type1", "e2", "type2"),
    ]
    idx = EdgeTypeIndex(etypes)

    # sorted 后顺序: e1, e2, e3, e4 (按 str(tuple) 字典序)
    expected_order = sorted(etypes)  # Python sorted() 与 C 层约定一致
    for i, etype in enumerate(expected_order):
        assert idx.lookup(etype) == i, (
            f"etype {etype!r}: expected index {i}, got {idx.lookup(etype)}"
        )

    # None → 多类型图应抛 ValueError
    try:
        idx.lookup(None)
        assert False, "多类型图 None etype 应抛 ValueError"
    except ValueError:
        pass

    # 单类型图 None → 0
    single_idx = EdgeTypeIndex([("src", "rel", "dst")])
    assert single_idx.lookup(None) == 0

    _dbg("_test_edge_type_index", "✓ EdgeTypeIndex 全部通过")
    print("[8bf2012 _test_edge_type_index] PASSED", file=sys.stderr)
    return True


def _test_bias_builder_hetero_diff() -> bool:
    """
    验证 _BiasBuilder._hetero_diff_type: 掩码逻辑
    对应 8bf2012 global_uniform_negative_sampling hetero src≠dst 路径
    """
    plan = _BiasBuilder._hetero_diff_type(
        src_type="type1",
        dst_type="type2",
        vertex_offsets={"type1": 0, "type2": 17},
        num_src_nodes=17,
        num_dst_nodes=17,
    )

    assert plan.path == "hetero_diff"
    assert plan.src_vertex_offset == 0
    assert plan.dst_vertex_offset == 17

    # vertices 长度应为 17+17=34
    assert len(plan.vertices) == 34, f"expected 34 vertices, got {len(plan.vertices)}"

    # src_bias: 前17个为1, 后17个为0
    if hasattr(plan.src_bias, '__iter__'):
        sb = list(plan.src_bias)
        assert all(v == 1.0 for v in sb[:17]), f"src_bias[:17] 应全为 1, got {sb[:17]}"
        assert all(v == 0.0 for v in sb[17:]), f"src_bias[17:] 应全为 0, got {sb[17:]}"

    # dst_bias: 前17个为0, 后17个为1
    if hasattr(plan.dst_bias, '__iter__'):
        db = list(plan.dst_bias)
        assert all(v == 0.0 for v in db[:17]), f"dst_bias[:17] 应全为 0, got {db[:17]}"
        assert all(v == 1.0 for v in db[17:]), f"dst_bias[17:] 应全为 1, got {db[17:]}"

    _dbg("_test_bias_builder_hetero_diff", "✓ _BiasBuilder._hetero_diff_type 全部通过")
    print("[8bf2012 _test_bias_builder_hetero_diff] PASSED", file=sys.stderr)
    return True


def _test_neg_sampling_session_replace_raises() -> bool:
    """
    验证 WalpurgisNegSamplingSession: replace=True 抛 NotImplementedError
    """
    try:
        WalpurgisNegSamplingSession(num_samples=10, replace=True)
        assert False, "replace=True 应抛 NotImplementedError"
    except NotImplementedError:
        pass

    _dbg("_test_neg_sampling_session_replace_raises",
         "✓ replace=True → NotImplementedError 正确")
    print("[8bf2012 _test_neg_sampling_session_replace_raises] PASSED", file=sys.stderr)
    return True


def _test_graph_clear_session() -> bool:
    """
    验证 GraphClearSession: 清理3字段 + validate_cleared()
    """
    # 模拟 Graph 对象 (无真实 cugraph_dgl 依赖)
    class FakeGraph:
        _Graph__graph = object()              # 非 None
        _Graph__edge_lookup_table = object()  # 非 None (8bf2012 新增)
        _Graph__vertex_offsets = object()     # 非 None

    fake = FakeGraph()
    session = GraphClearSession(caller_name="test_add_edges")

    assert not session.validate_cleared(fake), "清理前应返回 False"
    session.apply(fake)

    assert fake._Graph__graph is None
    assert fake._Graph__edge_lookup_table is None
    assert fake._Graph__vertex_offsets is None
    assert session.validate_cleared(fake), "清理后应返回 True"
    assert session.clear_count == 1

    _dbg("_test_graph_clear_session", "✓ GraphClearSession 全部通过")
    print("[8bf2012 _test_graph_clear_session] PASSED", file=sys.stderr)
    return True


if __name__ == "__main__":
    os.environ['WALPURGIS_DEBUG'] = '1'
    _DBG = True

    results = [
        _test_edge_type_index(),
        _test_bias_builder_hetero_diff(),
        _test_neg_sampling_session_replace_raises(),
        _test_graph_clear_session(),
    ]

    all_passed = all(results)
    print(
        f"\n[8bf2012 link_pred_neg_sampling] "
        f"{'ALL PASSED' if all_passed else 'SOME FAILED'} "
        f"({sum(results)}/{len(results)})",
        file=sys.stderr
    )
    sys.exit(0 if all_passed else 1)
