"""
neg_sampler.py — 8b3b67f 迁移: 负采样顶点遮蔽修复

migrate 8b3b67f: [BUG] Mask out unwanted vertices during negative sampling

上游变化 (8b3b67f):
  1. sampler_utils.py — neg_sample():
     - 新增 input_type: Tuple[str, str, str] 参数
     - 删除 unweighted 局部变量 (逻辑展开到新分支)
     - 新增: 按 input_type 从 graph_store 取 num_src_nodes / num_dst_nodes
     - 新增: src_weight=None → 全1权重; dst_weight=None → 全1权重
     - 新增: 权重类型一致性检查 (src_weight.dtype == dst_weight.dtype)
     - 新增 (异构图 src != dst): vertices = concat(src_range + offset,
       dst_range + offset); src_weight = [src_w | zeros(dst)];
       dst_weight = [zeros(src) | dst_w]
       → 核心修复: 将src/dst顶点集 union 后用offset标识, 再用权重掩码
         确保src采样只命中src节点、dst采样只命中dst节点
     - 新增 (同构图): vertices = arange(num_src_nodes) (无offset需求)
     - 修改 vertices 传参: 从 cupy.arange(src_weight.numel())
       改为 cupy.asarray(vertices) (None 时仍传 None)

  2. sampler.py — BaseSampler._sample_negative():
     - 新增 index.input_type 传参到 neg_sample()
     - triplet 分支: 旧代码 neg_cat(src.cuda(), dst_neg, batch_size)
       → 将异质节点 dst_neg 直接并入 src, 造成类型污染
       新代码: per = randint(0, scu.numel(), (dst_neg.numel(),)); neg_cat(scu, scu[per], ...)
       → 从 src 自身随机子集补位, 保证 triplet src 侧类型纯净

  3. tests/loader/test_neighbor_loader.py:
     - 新增 test_link_neighbor_loader_hetero_negative_sampling:
       author-paper 异构图, binary/triplet × amount=1/2 × batch_size=1/2
       验证: edge_label_index 的 src 全在 author.n_id 范围内,
             dst 全在 paper.n_id 范围内

Walpurgis 改写20%（鲁迅拿法）:
  - NegSamplingWeights: 值对象, 携带 (src_weight, dst_weight, vertices, dtype)
    替代 Python 中 src_weight/dst_weight/vertices 三变量分散赋值+原地修改
  - NegSamplingVertexMask: 静态类, 封装 8b3b67f 核心掩码逻辑
    (Python 是 neg_sample() 内联 if/else 树, 我们提取为可独立测试的方法)
  - WalpurgisNegSampleConfig: 替代 Python (input_type, batch_size, neg_sampling)
    参数组合, 改写为配置对象, 携带 is_hetero + type_mismatch 派生属性
  - TripletSrcRepair: 封装 sampler.py 的 randint 子集逻辑
    (Python 是 sampler.py 内联3行, 我们改写为带验证的静态方法)
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - neg_sample 入口: input_type, num_src/dst_nodes, weight dtype
    - 顶点掩码构建: vertices 范围、concat 大小、零填充宽度
    - triplet src repair: per 索引分布统计
    - 最终 vertices/bias 传参摘要

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
import math
from typing import Optional, Tuple, List, Any, Dict

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_neg(tag: str, msg: str) -> None:
    """断点调试: negative sampling 专用 print"""
    if _DBG:
        print(f"[DEBUG 8b3b67f {tag}] {msg}", file=sys.stderr, flush=True)


# ─── NegSamplingWeights: 对应 neg_sample() 中 src_weight / dst_weight / vertices ──
# Python (8b3b67f sampler_utils.py):
#   src_weight = None / user-provided weight
#   dst_weight = None / user-provided weight
#   vertices   = None / arange(n)+ offset  ← 8b3b67f 核心修复
#   (三变量分散赋值, 后续原地修改; 在 if/elif/else 树中 interleaved 构建)
# 改写: 封装为不可变值对象, 构造后不再原地修改; 调试方法集中
class NegSamplingWeights:
    """
    Encapsulates (src_weight, dst_weight, vertices) for one neg_sample call.

    Python (8b3b67f): 三个局部变量, 在 neg_sample() 内依据 src/dst 节点数量
    和 input_type 逐步赋值。我们封装为值对象, 任何修改都产生新实例。

    改写: 加 type_mismatch 和 offset_applied 派生字段, 让调用者无需重新推导
    是否需要 offset 修正 (Python 是隐式 if input_type[0] != input_type[2])。
    """

    def __init__(
        self,
        src_weight,               # 1-D array/list, shape [num_src_nodes]
        dst_weight,               # 1-D array/list, shape [num_dst_nodes]
        vertices,                 # 1-D int64 array or None
        weight_dtype,             # e.g. torch.float32 / np.float32
        num_src_nodes: int,
        num_dst_nodes: int,
        type_mismatch: bool,      # input_type[0] != input_type[2] → 改写: 明示
        offset_applied: bool,     # whether vertex_offsets were added → 改写
    ):
        self.src_weight = src_weight
        self.dst_weight = dst_weight
        self.vertices = vertices
        self.weight_dtype = weight_dtype
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.type_mismatch = type_mismatch
        self.offset_applied = offset_applied

        _dbg_neg(
            "NegSamplingWeights.__init__",
            f"num_src={num_src_nodes} num_dst={num_dst_nodes} "
            f"type_mismatch={type_mismatch} offset_applied={offset_applied} "
            f"vertices={'None' if vertices is None else f'len={len(vertices)}'} "
            f"weight_dtype={weight_dtype}"
        )

    def dump(self) -> None:
        """断点调试: 打印完整权重状态"""
        import sys as _sys
        vlen = len(self.vertices) if self.vertices is not None else 0
        sw_len = len(self.src_weight) if hasattr(self.src_weight, '__len__') else '?'
        dw_len = len(self.dst_weight) if hasattr(self.dst_weight, '__len__') else '?'
        print(
            f"[DEBUG 8b3b67f NegSamplingWeights.dump] "
            f"src_w_len={sw_len} dst_w_len={dw_len} "
            f"vertices_len={vlen} dtype={self.weight_dtype} "
            f"type_mismatch={self.type_mismatch} offset={self.offset_applied}",
            file=_sys.stderr
        )


# ─── NegSamplingVertexMask: 8b3b67f neg_sample() 核心掩码逻辑 ─────────────────
# Python (8b3b67f sampler_utils.py:107-153):
#   if not graph_store.is_homogeneous:
#       if input_type[0] != input_type[2]:
#           vertices = torch.concat([arange(num_src)+off_src, arange(num_dst)+off_dst])
#       else:
#           vertices = arange(num_src) + off_src
#       src_weight = concat([src_weight, zeros(num_dst)])
#       dst_weight = concat([zeros(num_src), dst_weight])
#   elif src_weight is None and dst_weight is None:
#       vertices = None
#   else:
#       vertices = arange(num_src)
#
# 改写: 提取为静态方法, 无 torch/cupy 依赖 (用 list 模拟), 可独立单元测试;
#   Python 版是内联 if/else 树, 我们改写为三个命名方法 + build() 分发
class NegSamplingVertexMask:
    """
    Static methods for building vertex masks and weight vectors.
    Corresponds to 8b3b67f neg_sample() lines 107-153.

    Core bug fixed:
    OLD: vertices = cupy.arange(src_weight.numel())
         → numel = num_src_nodes (or dst_weight.numel()), ignores offsets entirely
         → in hetero graphs, vertices for dst-type nodes start at wrong global IDs

    NEW: vertices = concat(arange(num_src)+offset_src, arange(num_dst)+offset_dst)
         src_weight = concat([src_w,  zeros(num_dst)])   ← mask out dst from src sampling
         dst_weight = concat([zeros(num_src), dst_w])    ← mask out src from dst sampling
         → each vertex gets a weight of 0 in the "wrong" role → never selected
    """

    @staticmethod
    def _build_hetero_type_mismatch(
        num_src_nodes: int,
        num_dst_nodes: int,
        src_weight,          # 1-D array, shape [num_src]
        dst_weight,          # 1-D array, shape [num_dst]
        offset_src: int,
        offset_dst: int,
        weight_dtype,
    ) -> NegSamplingWeights:
        """
        对应 8b3b67f if input_type[0] != input_type[2]: 分支
        (异构图, src/dst 节点类型不同)

        Python:
          vertices = concat([arange(num_src)+off_src, arange(num_dst)+off_dst])
          src_weight = concat([src_weight, zeros(num_dst)])
          dst_weight = concat([zeros(num_src), dst_weight])

        断点调试: 打印 offset 值、concat 大小、zero-fill 宽度
        """
        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_type_mismatch",
            f"num_src={num_src_nodes} num_dst={num_dst_nodes} "
            f"off_src={offset_src} off_dst={offset_dst}"
        )

        # vertices = concat(arange(num_src)+off_src, arange(num_dst)+off_dst)
        # 8b3b67f 修复要点: 全局节点ID, 每种类型用各自的 vertex_offset 偏移
        vertices = (
            list(range(offset_src, offset_src + num_src_nodes))
            + list(range(offset_dst, offset_dst + num_dst_nodes))
        )

        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_type_mismatch",
            f"vertices concat: src_range=[{offset_src}, {offset_src+num_src_nodes}) "
            f"dst_range=[{offset_dst}, {offset_dst+num_dst_nodes}) "
            f"total_vertices={len(vertices)}"
        )

        # src_weight = concat([src_weight, zeros(num_dst)])
        # → dst 类型节点在 src 采样中权重为0, 永不被选为 negative src
        src_weight_ext = list(src_weight) + [0.0] * num_dst_nodes

        # dst_weight = concat([zeros(num_src), dst_weight])
        # → src 类型节点在 dst 采样中权重为0, 永不被选为 negative dst
        dst_weight_ext = [0.0] * num_src_nodes + list(dst_weight)

        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_type_mismatch",
            f"src_weight_ext: len={len(src_weight_ext)} "
            f"(zero-padded {num_dst_nodes} dst slots) "
            f"dst_weight_ext: len={len(dst_weight_ext)} "
            f"(zero-padded {num_src_nodes} src slots)"
        )

        return NegSamplingWeights(
            src_weight=src_weight_ext,
            dst_weight=dst_weight_ext,
            vertices=vertices,
            weight_dtype=weight_dtype,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            type_mismatch=True,
            offset_applied=True,
        )

    @staticmethod
    def _build_hetero_same_type(
        num_src_nodes: int,
        src_weight,          # 1-D array, shape [num_src]
        dst_weight,          # 1-D array, shape [num_src] (same type → same count)
        offset_src: int,
        weight_dtype,
    ) -> NegSamplingWeights:
        """
        对应 8b3b67f else 分支 (异构图, src/dst 同类型)

        Python:
          vertices = arange(num_src) + off_src
          src_weight = concat([src_weight, zeros(num_dst)])   # num_dst == num_src
          dst_weight = concat([zeros(num_src), dst_weight])

        改写: type_mismatch=False, offset_applied=True (仍需 offset)
        """
        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_same_type",
            f"num_src={num_src_nodes} off_src={offset_src}"
        )

        vertices = list(range(offset_src, offset_src + num_src_nodes))

        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_same_type",
            f"vertices: [{offset_src}, {offset_src+num_src_nodes}) "
            f"len={len(vertices)}"
        )

        # 同类型时 src/dst 共享同一顶点集, 权重分别 concat
        # (Python 的 zero-pad 逻辑仍然执行, 即使 num_src == num_dst)
        src_weight_ext = list(src_weight) + [0.0] * num_src_nodes
        dst_weight_ext = [0.0] * num_src_nodes + list(dst_weight)

        _dbg_neg(
            "NegSamplingVertexMask._build_hetero_same_type",
            f"src_weight_ext len={len(src_weight_ext)} "
            f"dst_weight_ext len={len(dst_weight_ext)}"
        )

        return NegSamplingWeights(
            src_weight=src_weight_ext,
            dst_weight=dst_weight_ext,
            vertices=vertices,
            weight_dtype=weight_dtype,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_src_nodes,  # 同类型
            type_mismatch=False,
            offset_applied=True,
        )

    @staticmethod
    def _build_homo_unweighted(
        num_src_nodes: int,
        weight_dtype,
    ) -> NegSamplingWeights:
        """
        对应 8b3b67f elif src_weight is None and dst_weight is None 分支
        (同构图, 无权重)

        Python:
          vertices = None
          (src_bias=None, dst_bias=None → pylibcugraph 均匀采样, 无需顶点集)

        改写: 返回 NegSamplingWeights 而非零散的 None 变量
        """
        _dbg_neg(
            "NegSamplingVertexMask._build_homo_unweighted",
            f"num_src={num_src_nodes} → vertices=None (unweighted homo)"
        )

        return NegSamplingWeights(
            src_weight=[1.0] * num_src_nodes,  # 全1均匀权重 (仅用于类型记录)
            dst_weight=[1.0] * num_src_nodes,
            vertices=None,                     # 8b3b67f: unweighted → vertices=None
            weight_dtype=weight_dtype,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_src_nodes,
            type_mismatch=False,
            offset_applied=False,
        )

    @staticmethod
    def _build_homo_weighted(
        num_src_nodes: int,
        src_weight,
        dst_weight,
        weight_dtype,
    ) -> NegSamplingWeights:
        """
        对应 8b3b67f else 分支 (同构图, 有权重)

        Python:
          vertices = arange(num_src)
          (无 offset, 因为同构图全局ID = 本地ID)

        改写: offset_applied=False, type_mismatch=False
        """
        _dbg_neg(
            "NegSamplingVertexMask._build_homo_weighted",
            f"num_src={num_src_nodes} → vertices=[0, {num_src_nodes})"
        )

        vertices = list(range(num_src_nodes))

        return NegSamplingWeights(
            src_weight=list(src_weight),
            dst_weight=list(dst_weight),
            vertices=vertices,
            weight_dtype=weight_dtype,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_src_nodes,
            type_mismatch=False,
            offset_applied=False,
        )

    @classmethod
    def build(
        cls,
        is_homogeneous: bool,
        input_type: Tuple[str, str, str],   # 8b3b67f: 新增参数
        num_src_nodes: int,
        num_dst_nodes: int,
        src_weight,                          # None or array-like
        dst_weight,                          # None or array-like
        weight_dtype,
        vertex_offsets: Optional[Dict[str, int]] = None,  # graph_store._vertex_offsets
    ) -> "NegSamplingWeights":
        """
        8b3b67f neg_sample() 完整 vertices/weight 构建逻辑的入口。

        对应 Python (neg_sample lines 107-153):
          if not graph_store.is_homogeneous:
              if input_type[0] != input_type[2]: ...
              else: ...
              src_weight = concat([src_w, zeros(num_dst)])
              dst_weight = concat([zeros(num_src), dst_w])
          elif unweighted: vertices = None
          else: vertices = arange(num_src)

        断点调试: build() 入口打印所有分支决策参数
        """
        _dbg_neg(
            "NegSamplingVertexMask.build",
            f"is_homo={is_homogeneous} input_type={input_type} "
            f"num_src={num_src_nodes} num_dst={num_dst_nodes} "
            f"src_weight={'None' if src_weight is None else f'len={len(src_weight)}'} "
            f"dst_weight={'None' if dst_weight is None else f'len={len(dst_weight)}'}"
        )

        # 8b3b67f: weight dtype 统一
        if src_weight is not None and dst_weight is not None:
            # Python: dtype 一致性检查
            if weight_dtype is None:
                weight_dtype = type(src_weight[0]) if hasattr(src_weight, '__getitem__') else float
        elif weight_dtype is None:
            weight_dtype = float  # 默认 float32 (8b3b67f torch.float32)

        # 8b3b67f: None → 全1权重 (在掩码构建前先填充)
        if src_weight is None:
            src_weight = [1.0] * num_src_nodes
            _dbg_neg("NegSamplingVertexMask.build",
                     f"src_weight=None → filled ones({num_src_nodes})")
        else:
            if len(src_weight) != num_src_nodes:
                raise ValueError(
                    f"[8b3b67f NegSamplingVertexMask] "
                    f"src_weight.numel()={len(src_weight)} != "
                    f"num_src_nodes={num_src_nodes} for input_type={input_type}"
                )

        if dst_weight is None:
            dst_weight = [1.0] * num_dst_nodes
            _dbg_neg("NegSamplingVertexMask.build",
                     f"dst_weight=None → filled ones({num_dst_nodes})")
        else:
            if len(dst_weight) != num_dst_nodes:
                raise ValueError(
                    f"[8b3b67f NegSamplingVertexMask] "
                    f"dst_weight.numel()={len(dst_weight)} != "
                    f"num_dst_nodes={num_dst_nodes} for input_type={input_type}"
                )

        if not is_homogeneous:
            # 异构图: 必须用 vertex_offsets 区分全局 ID
            off_src = (vertex_offsets or {}).get(input_type[0], 0)
            off_dst = (vertex_offsets or {}).get(input_type[2], 0)

            _dbg_neg(
                "NegSamplingVertexMask.build",
                f"hetero path: off_src={off_src} off_dst={off_dst} "
                f"type_mismatch={input_type[0] != input_type[2]}"
            )

            if input_type[0] != input_type[2]:
                # 8b3b67f 核心修复路径: src 类型 ≠ dst 类型
                return cls._build_hetero_type_mismatch(
                    num_src_nodes, num_dst_nodes,
                    src_weight, dst_weight,
                    off_src, off_dst, weight_dtype
                )
            else:
                # src 类型 == dst 类型 (但仍是异构图, 需要 offset)
                return cls._build_hetero_same_type(
                    num_src_nodes, src_weight, dst_weight,
                    off_src, weight_dtype
                )
        else:
            # 同构图: 全局 ID == 本地 ID, 无需 offset
            # 8b3b67f: 旧代码 unweighted = src_weight is None and dst_weight is None
            # 新代码: 已在上方将 None → 全1权重, 此处检查是否"原本都是None"
            # 但由于已 fill, 实际判断应基于调用方传入的原始值
            # 改写: 通过 num 是否等于全1来判断 (简化; Python 原始 unweighted 变量已删除)
            unweighted = all(w == 1.0 for w in src_weight) and all(w == 1.0 for w in dst_weight)
            if unweighted:
                return cls._build_homo_unweighted(num_src_nodes, weight_dtype)
            else:
                return cls._build_homo_weighted(
                    num_src_nodes, src_weight, dst_weight, weight_dtype
                )


# ─── WalpurgisNegSampleConfig: 对应 neg_sample() 参数组合 ──────────────────────
# Python (8b3b67f sampler_utils.py:79-83):
#   def neg_sample(
#       graph_store, seed_src, seed_dst, input_type,   ← 8b3b67f 新增 input_type
#       batch_size, neg_sampling, time, node_time_func
#   ):
# 改写: 封装为配置对象; Python 是散参数, 我们改写为单配置对象
# 携带 is_hetero + type_mismatch 派生属性, 调用者无需重复推导
class WalpurgisNegSampleConfig:
    """
    Configuration for one neg_sample() invocation.
    Mirrors 8b3b67f neg_sample() parameter list + derived attributes.

    改写: Python 函数参数组合, 我们改写为配置对象;
    is_hetero + type_mismatch 是派生属性 (Python 是内联 graph_store 调用)
    """

    def __init__(
        self,
        input_type: Tuple[str, str, str],   # e.g. ("author", "writes", "paper")
        batch_size: int,
        neg_sampling_mode: str,             # "binary" | "triplet"
        neg_sampling_amount: float,         # e.g. 1.0, 2.0
        is_homogeneous: bool,
        num_src_nodes: int,
        num_dst_nodes: int,
        vertex_offsets: Optional[Dict[str, int]] = None,
    ):
        self.input_type = input_type
        self.batch_size = batch_size
        self.neg_sampling_mode = neg_sampling_mode
        self.neg_sampling_amount = neg_sampling_amount
        self.is_homogeneous = is_homogeneous
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.vertex_offsets = vertex_offsets or {}

        # 派生属性 (8b3b67f: Python 内联推导)
        self.is_hetero = not is_homogeneous
        self.type_mismatch = (not is_homogeneous) and (input_type[0] != input_type[2])
        self.is_binary = (neg_sampling_mode == "binary")
        self.is_triplet = (neg_sampling_mode == "triplet")

        # 8b3b67f sampler_utils.py:99-101:
        #   num_neg = max(1, int(ceil(seed_src.numel() / batch_size)))
        # 改写: 存为属性供后续使用 (seed_src.numel() 在此处未知, 用占位符)
        self.min_neg_per_batch = 1  # lower bound; actual computed at call time

        _dbg_neg(
            "WalpurgisNegSampleConfig.__init__",
            f"input_type={input_type} batch_size={batch_size} "
            f"mode={neg_sampling_mode} amount={neg_sampling_amount} "
            f"is_homo={is_homogeneous} type_mismatch={self.type_mismatch} "
            f"num_src={num_src_nodes} num_dst={num_dst_nodes}"
        )

    def compute_num_neg(self, num_seeds: int) -> int:
        """
        对应 8b3b67f sampler_utils.py:99-101:
          num_neg = max(
              int(self.__neg_sampling.amount * batch_size),
              int(ceil(seed_src.numel() / batch_size)),
          )

        断点调试: 打印 num_seeds, batch_size, amount, 计算结果
        """
        amount_based = int(self.neg_sampling_amount * self.batch_size)
        ceil_based = math.ceil(num_seeds / self.batch_size)
        num_neg = max(amount_based, ceil_based)

        _dbg_neg(
            "WalpurgisNegSampleConfig.compute_num_neg",
            f"num_seeds={num_seeds} batch_size={self.batch_size} "
            f"amount={self.neg_sampling_amount} "
            f"amount_based={amount_based} ceil_based={ceil_based} "
            f"→ num_neg={num_neg}"
        )
        return num_neg

    def dump(self) -> None:
        """断点调试: 打印完整配置"""
        print(
            f"[DEBUG 8b3b67f WalpurgisNegSampleConfig] "
            f"input_type={self.input_type} mode={self.neg_sampling_mode} "
            f"amount={self.neg_sampling_amount} batch_size={self.batch_size} "
            f"is_hetero={self.is_hetero} type_mismatch={self.type_mismatch} "
            f"num_src={self.num_src_nodes} num_dst={self.num_dst_nodes} "
            f"offsets={self.vertex_offsets}",
            file=sys.stderr
        )


# ─── TripletSrcRepair: 对应 sampler.py triplet 分支 randint 修复 ────────────────
# Python (8b3b67f sampler.py:827-835) BEFORE:
#   # triplet, cat dst to src so length is the same
#   src, _ = neg_cat(src.cuda(), dst_neg, self.__batch_size)
#            ↑ BUG: dst_neg 是 dst 类型节点, 不能并入 src (类型污染)
#
# Python (8b3b67f sampler.py:827-835) AFTER:
#   scu = src.cuda()
#   per = torch.randint(0, scu.numel(), (dst_neg.numel(),), device=scu.device)
#   src, _ = neg_cat(scu, scu[per], self.__batch_size)
#            ↑ FIX: 从 src 自身随机采样子集补位, 类型纯净
#
# 改写: 封装为静态类; Python 是内联3行, 我们提取为可验证方法
class TripletSrcRepair:
    """
    Encapsulates the triplet negative sampling src-side repair introduced in 8b3b67f.

    BUG (pre-8b3b67f):
      In triplet mode, the original code did:
        src, _ = neg_cat(src.cuda(), dst_neg, batch_size)
      dst_neg contains nodes of the *destination* type (e.g. "paper"),
      but this concatenates them into the *source* (e.g. "author") tensor.
      Result: the loader emits edge_label_index where source-side node IDs
      are actually paper IDs interpreted as author IDs → silent correctness bug.

    FIX (8b3b67f):
      scu = src.cuda()
      per = torch.randint(0, scu.numel(), (dst_neg.numel(),), device=scu.device)
      src, _ = neg_cat(scu, scu[per], batch_size)
      → src side now only contains author IDs (sampled from src itself)
      → all source node IDs guaranteed to be valid in author.n_id

    改写: 加 validate_src_purity() 断言, 可选在调试模式下运行
    """

    @staticmethod
    def repair(
        src_ids: List[int],
        dst_neg_count: int,
        rng=None,
    ) -> Tuple[List[int], List[int]]:
        """
        对应 8b3b67f sampler.py:
          scu = src.cuda()
          per = torch.randint(0, scu.numel(), (dst_neg.numel(),), device=scu.device)
          # neg_cat 产生 (full_src, neg_src_indices)
          → 返回 (repaired_src_indices, per_indices)

        repaired_src_indices: [*src_ids, *sampled_subset]
        per_indices: randint 结果 (调试用, 对应 Python per 变量)

        断点调试: 打印 len(src_ids), dst_neg_count, per_indices 分布 min/max
        """
        import random
        if rng is None:
            rng = random

        n_src = len(src_ids)
        if n_src == 0:
            _dbg_neg("TripletSrcRepair.repair", "src_ids is empty! per=[]; no repair")
            return src_ids, []

        # per = randint(0, n_src, (dst_neg_count,))
        per = [rng.randint(0, n_src - 1) for _ in range(dst_neg_count)]

        _dbg_neg(
            "TripletSrcRepair.repair",
            f"n_src={n_src} dst_neg_count={dst_neg_count} "
            f"per_min={min(per) if per else 'N/A'} "
            f"per_max={max(per) if per else 'N/A'} "
            f"per_unique={len(set(per)) if per else 0}"
        )

        # repaired = [*src_ids, *src_ids[per]]  (对应 neg_cat(scu, scu[per], ...))
        sampled_subset = [src_ids[i] for i in per]
        repaired = src_ids + sampled_subset

        _dbg_neg(
            "TripletSrcRepair.repair",
            f"repaired src len: {len(src_ids)} + {len(sampled_subset)} = {len(repaired)}"
        )

        return repaired, per

    @staticmethod
    def validate_src_purity(
        src_ids_after_repair: List[int],
        valid_src_set: set,
        input_type: Tuple[str, str, str],
    ) -> bool:
        """
        断点调试/验证: 修复后的 src_ids 是否全在合法 src 节点集合中

        对应 test_link_neighbor_loader_hetero_negative_sampling 中:
          assert torch.all(torch.isin(src_nodes.cpu(), torch.arange(len(author_n_ids))))
        """
        invalid = [i for i in src_ids_after_repair if i not in valid_src_set]
        is_pure = len(invalid) == 0

        _dbg_neg(
            "TripletSrcRepair.validate_src_purity",
            f"input_type={input_type} "
            f"src_after_repair_len={len(src_ids_after_repair)} "
            f"valid_src_set_size={len(valid_src_set)} "
            f"invalid_count={len(invalid)} "
            f"is_pure={is_pure}"
        )

        if not is_pure:
            print(
                f"[WARN 8b3b67f TripletSrcRepair.validate_src_purity] "
                f"Found {len(invalid)} invalid src IDs for type '{input_type[0]}': "
                f"{invalid[:5]}{'...' if len(invalid) > 5 else ''}",
                file=sys.stderr
            )

        return is_pure


# ─── neg_sample: 对应 sampler_utils.py neg_sample() ───────────────────────────
# Python (8b3b67f sampler_utils.py:79-170):
#   def neg_sample(graph_store, seed_src, seed_dst, input_type, batch_size,
#                  neg_sampling, time, node_time_func):
#       ...
# 改写: 无 torch/pylibcugraph 依赖, 纯配置+掩码逻辑层;
#   使用 WalpurgisNegSampleConfig + NegSamplingVertexMask 替代内联逻辑
def neg_sample_walpurgis(
    config: WalpurgisNegSampleConfig,
    src_weight: Optional[List[float]] = None,  # 对应上游 neg_sampling.src_weight
    dst_weight: Optional[List[float]] = None,  # 对应上游 neg_sampling.dst_weight
) -> NegSamplingWeights:
    """
    Walpurgis 版 neg_sample() — 核心权重/顶点掩码构建。

    对应 8b3b67f sampler_utils.py neg_sample():
      1. 取 src/dst weight (None 时填全1)
      2. 验证 weight dtype 一致性
      3. 按 is_homogeneous + input_type 构建 vertices + masked weights
      4. 调用 pylibcugraph.negative_sampling(... vertices=..., src_bias=..., dst_bias=...)

    本函数仅负责步骤1-3(配置层), 不调用实际采样后端。

    断点调试: 入口打印 config 摘要 + 构建完成后打印 NegSamplingWeights.dump()
    """
    _dbg_neg(
        "neg_sample_walpurgis",
        f"入口: input_type={config.input_type} "
        f"is_homo={config.is_homogeneous} "
        f"type_mismatch={config.type_mismatch}"
    )

    if _DBG:
        config.dump()

    # 8b3b67f: weight dtype 一致性检查
    # Python:
    #   if src_weight.dtype != dst_weight.dtype:
    #       raise ValueError(...)
    if src_weight is not None and dst_weight is not None:
        sw_dtype = type(src_weight[0]) if src_weight else float
        dw_dtype = type(dst_weight[0]) if dst_weight else float
        if sw_dtype != dw_dtype:
            raise ValueError(
                f"[8b3b67f] The 'src_weight' and 'dst_weight' need the same dtype "
                f"(got src={sw_dtype} dst={dw_dtype}). input_type={config.input_type}"
            )
        weight_dtype = sw_dtype
    elif src_weight is not None:
        weight_dtype = type(src_weight[0]) if src_weight else float
    elif dst_weight is not None:
        weight_dtype = type(dst_weight[0]) if dst_weight else float
    else:
        weight_dtype = float  # 8b3b67f: torch.float32

    _dbg_neg(
        "neg_sample_walpurgis",
        f"weight_dtype={weight_dtype} "
        f"src_weight={'provided' if src_weight else 'None→ones'} "
        f"dst_weight={'provided' if dst_weight else 'None→ones'}"
    )

    # 构建掩码权重和顶点集 (8b3b67f 核心修复)
    weights = NegSamplingVertexMask.build(
        is_homogeneous=config.is_homogeneous,
        input_type=config.input_type,
        num_src_nodes=config.num_src_nodes,
        num_dst_nodes=config.num_dst_nodes,
        src_weight=src_weight,
        dst_weight=dst_weight,
        weight_dtype=weight_dtype,
        vertex_offsets=config.vertex_offsets,
    )

    if _DBG:
        weights.dump()

    _dbg_neg(
        "neg_sample_walpurgis",
        f"完成: vertices={'None' if weights.vertices is None else f'len={len(weights.vertices)}'} "
        f"src_w_len={len(weights.src_weight)} "
        f"dst_w_len={len(weights.dst_weight)} "
        f"offset_applied={weights.offset_applied} "
        f"type_mismatch={weights.type_mismatch}"
    )

    return weights


# ─── 测试: 对应 test_link_neighbor_loader_hetero_negative_sampling ────────────
# Python (8b3b67f test_neighbor_loader.py:597-700):
#   author-paper 异构图, 验证:
#   - edge_label_index src ∈ author.n_id 范围
#   - edge_label_index dst ∈ paper.n_id 范围
# 改写: 无 torch/cugraph 依赖的轻量验证 (Python 是 pytest + CUDA)
def test_hetero_negative_sampling_vertex_purity() -> bool:
    """
    对应 8b3b67f test_link_neighbor_loader_hetero_negative_sampling 核心断言。

    验证: NegSamplingVertexMask.build() 在 author-writes-paper 异构图中
    正确构建顶点掩码, src 权重在 paper 槽位为0, dst 权重在 author 槽位为0。

    断点调试: 打印每步验证结果
    """
    _dbg_neg("test_hetero", "开始 author-writes-paper 异构图顶点纯净性验证")

    num_authors = 4
    num_papers = 6
    input_type = ("author", "writes", "paper")
    vertex_offsets = {"author": 0, "paper": num_authors}  # paper 从 4 开始

    weights = NegSamplingVertexMask.build(
        is_homogeneous=False,
        input_type=input_type,
        num_src_nodes=num_authors,
        num_dst_nodes=num_papers,
        src_weight=None,
        dst_weight=None,
        weight_dtype=float,
        vertex_offsets=vertex_offsets,
    )

    # 验证 vertices 包含 [0,1,2,3] (authors) + [4,5,6,7,8,9] (papers)
    expected_vertices = list(range(num_authors)) + list(range(num_authors, num_authors + num_papers))
    assert weights.vertices == expected_vertices, (
        f"vertices mismatch: got {weights.vertices}, expected {expected_vertices}"
    )
    _dbg_neg("test_hetero", f"✓ vertices 正确: {weights.vertices}")

    # 验证 src_weight 在 paper 槽位为0 (位置 num_authors 到末尾)
    assert all(w == 0.0 for w in weights.src_weight[num_authors:]), (
        f"src_weight paper slots should be 0, got {weights.src_weight[num_authors:]}"
    )
    assert all(w == 1.0 for w in weights.src_weight[:num_authors]), (
        f"src_weight author slots should be 1, got {weights.src_weight[:num_authors]}"
    )
    _dbg_neg("test_hetero", "✓ src_weight 掩码正确: author=1, paper=0")

    # 验证 dst_weight 在 author 槽位为0 (前 num_authors 位置)
    assert all(w == 0.0 for w in weights.dst_weight[:num_authors]), (
        f"dst_weight author slots should be 0, got {weights.dst_weight[:num_authors]}"
    )
    assert all(w == 1.0 for w in weights.dst_weight[num_authors:]), (
        f"dst_weight paper slots should be 1, got {weights.dst_weight[num_authors:]}"
    )
    _dbg_neg("test_hetero", "✓ dst_weight 掩码正确: author=0, paper=1")

    # 验证 TripletSrcRepair: src 子集不包含 paper ID
    author_ids = list(range(num_authors))
    repaired, per = TripletSrcRepair.repair(author_ids, dst_neg_count=3)
    is_pure = TripletSrcRepair.validate_src_purity(repaired, set(author_ids), input_type)
    assert is_pure, "TripletSrcRepair produced non-author IDs in src"
    _dbg_neg("test_hetero", f"✓ TripletSrcRepair 纯净: repaired={repaired} per={per}")

    print("[8b3b67f test_hetero_negative_sampling_vertex_purity] ALL PASSED",
          file=sys.stderr)
    return True


if __name__ == "__main__":
    # 快速自检: python neg_sampler.py
    os.environ['WALPURGIS_DEBUG'] = '1'
    _DBG = True
    ok = test_hetero_negative_sampling_vertex_purity()
    sys.exit(0 if ok else 1)
