"""
disjoint_sampler.py — b25bc88 迁移: Disjoint Sampling 接口层

migrate b25bc88: Support Disjoint Sampling in cuGraph-PyG

上游变化 (b25bc88):
  1. neighbor_loader.py + link_neighbor_loader.py:
     - 删除: "Currently unsupported." docstring行
     - 删除: if disjoint: raise ValueError("Disjoint sampling is currently unsupported")
     - 新增: disjoint=disjoint 传入 DistributedNeighborSampler(...)
  2. distributed_sampler.py:
     - __init__ 新增 disjoint: bool = False 参数
     - sample_kwargs dict 新增 "disjoint_sampling": disjoint
     - __calc_local_seeds_per_call 新增 disjoint: bool = False 参数
     - __calc_local_seeds_per_call: 修正 if len([x for x in fanout if x <= 0]) 判断顺序
       (原来在 heterogeneous 之前，移到 heterogeneous 之后 — 顺序bug fix)
     - __calc_local_seeds_per_call: disjoint=True 时 fanout_prod *= fanout[0]
       (disjoint 模式每个 seed 不去重，内存放大 fanout[0] 倍)
     - super().__init__ 调用改为关键字参数 local_seeds_per_call=
     - __calc_local_seeds_per_call 所有参数改为 keyword-only (*)
  3. tests/loader/test_neighbor_loader.py:
     - 新增 test_link_neighbor_loader_disjoint: 验证 disjoint=True/False 边数差异
     - 新增 test_neighbor_loader_disjoint: 验证 batch 结构 + n_id 集合
     - 新增 test_neighbor_loader_disjoint_batch_structure: 多 batch_size，
       验证 per-seed 子图互不相交 (tree_vertices 集合交集为空)

Walpurgis 改写20%（鲁迅拿法）:
  - 无 pylibcugraph/torch 依赖: DisjointSamplingConfig 封装所有 disjoint 采样配置
    替代 Python 的 bool flag + kwargs dict 分散模式
  - DisjointMemoryEstimator 替代 Python 内联 lambda reduce+fanout_prod 计算,
    改写为带单位注释的方法 + 调试输出; 并修正上游 bucket-order bug
  - WalpurgisDisjointSession 替代 DistributedNeighborSampler.__init__ 的
    disjoint 初始化路径, 改写为可序列化配置对象
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - 每次 make_disjoint_session 打印 fanout + memory_estimate
    - disjoint=True 时打印 fanout_prod amplification
    - per-seed 子图不相交验证 (validate_disjoint_batches) 打印 overlap 统计

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
from functools import reduce
from typing import List, Optional, Dict, Any, Tuple

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_disjoint(tag: str, msg: str) -> None:
    """断点调试: disjoint sampling 专用 print"""
    if _DBG:
        print(f"[DEBUG b25bc88 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── DisjointSamplingConfig: 对应 DistributedNeighborSampler 的 disjoint 配置 ──
# Python (b25bc88 distributed_sampler.py:772-789):
#   def __init__(self, ..., disjoint: bool = False, ...):
#       sample_kwargs = {
#           "compress_per_hop": compress_per_hop,
#           "compression": compression,
#           "with_replacement": with_replacement,
#           "disjoint_sampling": disjoint,         # ← b25bc88 新增
#       }
# 改写: 封装为 DisjointSamplingConfig 对象; Python 是 dict 散落在 __init__,
#       我们改写为单一配置对象, 所有 disjoint 相关参数集中管理
class DisjointSamplingConfig:
    """
    Encapsulates disjoint sampling configuration for one DistributedNeighborSampler.

    Python (b25bc88): disjoint 是 __init__ 参数, 存入 sample_kwargs["disjoint_sampling"]
    Walpurgis: 单一对象, 包含 disjoint flag + 衍生配置 + 调试方法

    改写: 加 is_amplified_memory 属性明示内存放大路径 (Python 是隐式 if disjoint: fanout_prod *= ...)
    """

    def __init__(
        self,
        disjoint: bool = False,
        compression: str = "COO",
        compress_per_hop: bool = False,
        with_replacement: bool = False,
    ):
        # b25bc88 distributed_sampler.py:788: "disjoint_sampling": disjoint
        self.disjoint = disjoint
        self.compression = compression
        self.compress_per_hop = compress_per_hop
        self.with_replacement = with_replacement

        # 改写: 显式标注内存放大标志 (Python 是 if disjoint: fanout_prod *= fanout[0] 内联)
        self.is_amplified_memory: bool = disjoint

        _dbg_disjoint(
            "DisjointSamplingConfig.__init__",
            f"disjoint={disjoint} compression={compression!r} "
            f"compress_per_hop={compress_per_hop} with_replacement={with_replacement} "
            f"is_amplified_memory={self.is_amplified_memory}"
        )

    def to_sample_kwargs(self) -> Dict[str, Any]:
        """
        对应 Python b25bc88 distributed_sampler.py:786-789:
            sample_kwargs = {
                "compress_per_hop": compress_per_hop,
                "compression": compression,
                "with_replacement": with_replacement,
                "disjoint_sampling": disjoint,
            }
        断点调试: 打印完整 sample_kwargs
        """
        kwargs = {
            "compress_per_hop": self.compress_per_hop,
            "compression": self.compression,
            "with_replacement": self.with_replacement,
            "disjoint_sampling": self.disjoint,   # b25bc88 新增的 key
        }
        _dbg_disjoint(
            "DisjointSamplingConfig.to_sample_kwargs",
            f"kwargs={kwargs}"
        )
        return kwargs

    def dump(self) -> None:
        """断点调试: 打印完整状态"""
        print(
            f"[DEBUG b25bc88 DisjointSamplingConfig] "
            f"disjoint={self.disjoint} compression={self.compression!r} "
            f"compress_per_hop={self.compress_per_hop} "
            f"with_replacement={self.with_replacement} "
            f"is_amplified_memory={self.is_amplified_memory}",
            file=sys.stderr
        )


# ─── DisjointMemoryEstimator: 对应 __calc_local_seeds_per_call ───────────────
# Python (b25bc88 distributed_sampler.py:836-872):
#   def __calc_local_seeds_per_call(*, local_seeds_per_call, heterogeneous, disjoint, num_edge_types):
#       fanout = self.__fanout
#       if local_seeds_per_call is None:
#           if heterogeneous:                          # 先处理 heterogeneous
#               if len(fanout) % num_edge_types != 0:
#                   raise ValueError(...)
#               num_hops = len(fanout) // num_edge_types
#               fanout = [max(fanout[h*num_edge_types:(h+1)*num_edge_types]) for h in range(num_hops)]
#           if len([x for x in fanout if x <= 0]) > 0:    # ← b25bc88 移到 heterogeneous 之后
#               return UNKNOWN_VERTICES_DEFAULT
#           fanout_prod = reduce(lambda x, y: x * y, fanout)
#           if disjoint:                               # ← b25bc88 新增
#               fanout_prod *= fanout[0]               # per-seed 不去重, 内存放大 fanout[0]
#           return int(BASE_VERTICES_PER_BYTE * total_memory / fanout_prod)
#
# Bug fix (b25bc88): 原先 "if len([x for x in fanout if x <= 0]) > 0" 在 heterogeneous
#   renormalization 之前 — 导致 heterogeneous+unknown_fanout 路径提前返回而跳过规范化
#   b25bc88 修正: 先 heterogeneous 规范化, 再检查 unknown_fanout
#
# 改写: 封装为 DisjointMemoryEstimator 类, 带单位注释 + 调试输出
class DisjointMemoryEstimator:
    """
    Memory estimation for local_seeds_per_call, with disjoint amplification.

    Constants mirror DistributedNeighborSampler:
      BASE_VERTICES_PER_BYTE: 类中保留上游原始值
      UNKNOWN_VERTICES_DEFAULT: 上游硬编码 fallback

    改写: Python 是 instance method, 我们改写为 pure function 封装在 class 里,
          便于单元测试 (无需构造完整 DistributedNeighborSampler)
    """

    # DistributedNeighborSampler 上游常量 (b25bc88 未改变这两个值)
    BASE_VERTICES_PER_BYTE: float = 1.0 / 100.0   # 原始值: 上游注释 "~100 bytes/vertex"
    UNKNOWN_VERTICES_DEFAULT: int = 32768           # fallback when fanout contains <=0

    @staticmethod
    def _normalize_hetero_fanout(
        fanout: List[int],
        num_edge_types: int,
    ) -> List[int]:
        """
        对应 b25bc88 distributed_sampler.py:849-856 heterogeneous 规范化:
            if len(fanout) % num_edge_types != 0:
                raise ValueError(...)
            num_hops = len(fanout) // num_edge_types
            fanout = [max(fanout[h*num_edge_types:(h+1)*num_edge_types]) for h in range(num_hops)]
        断点调试: 打印 before/after fanout
        """
        if len(fanout) % num_edge_types != 0:
            raise ValueError(
                f"Illegal fanout for {num_edge_types} edge types. "
                f"len(fanout)={len(fanout)} must be divisible by num_edge_types. "
                f"(b25bc88 distributed_sampler.py:851)"
            )
        num_hops = len(fanout) // num_edge_types
        normalized = [
            max(fanout[h * num_edge_types: (h + 1) * num_edge_types])
            for h in range(num_hops)
        ]
        _dbg_disjoint(
            "DisjointMemoryEstimator._normalize_hetero_fanout",
            f"raw_fanout={fanout} num_edge_types={num_edge_types} "
            f"num_hops={num_hops} normalized={normalized}"
        )
        return normalized

    @classmethod
    def calc_seeds_per_call(
        cls,
        fanout: List[int],
        total_memory_bytes: int,
        *,
        local_seeds_per_call: Optional[int] = None,
        heterogeneous: bool = False,
        disjoint: bool = False,
        num_edge_types: int = 1,
    ) -> int:
        """
        对应 Python b25bc88 __calc_local_seeds_per_call (keyword-only args, b25bc88 新增 *)

        b25bc88 修正的 bucket 顺序 (关键 bug fix):
          旧: unknown_fanout_check → heterogeneous_normalize → fanout_prod
          新: heterogeneous_normalize → unknown_fanout_check → fanout_prod

        断点调试: 打印 fanout_prod + disjoint 放大因子 + 最终 seeds_per_call
        """
        _dbg_disjoint(
            "DisjointMemoryEstimator.calc_seeds_per_call",
            f"fanout={fanout} total_memory_bytes={total_memory_bytes} "
            f"local_seeds_per_call={local_seeds_per_call} "
            f"heterogeneous={heterogeneous} disjoint={disjoint} "
            f"num_edge_types={num_edge_types}"
        )

        # 用户显式指定 → 直接返回
        if local_seeds_per_call is not None:
            _dbg_disjoint(
                "DisjointMemoryEstimator.calc_seeds_per_call",
                f"user-specified local_seeds_per_call={local_seeds_per_call}, skip estimation"
            )
            return local_seeds_per_call

        effective_fanout = list(fanout)

        # b25bc88 bucket 顺序修正: 先 heterogeneous 规范化
        # (旧代码在此之前检查 unknown_fanout, 导致 hetero+unknown 路径跳过规范化)
        if heterogeneous:
            effective_fanout = cls._normalize_hetero_fanout(effective_fanout, num_edge_types)

        # b25bc88 移后: 规范化之后再检查 unknown_fanout
        if any(x <= 0 for x in effective_fanout):
            _dbg_disjoint(
                "DisjointMemoryEstimator.calc_seeds_per_call",
                f"fanout contains <=0 value: {effective_fanout} → "
                f"return UNKNOWN_VERTICES_DEFAULT={cls.UNKNOWN_VERTICES_DEFAULT}"
            )
            return cls.UNKNOWN_VERTICES_DEFAULT

        # 计算 fanout 乘积
        fanout_prod = reduce(lambda x, y: x * y, effective_fanout)

        # b25bc88 disjoint 内存放大 (distributed_sampler.py:862-865):
        #   "Disjoint sampling produces unique (vertex, seed) pairs with no
        #    cross-seed deduplication, so memory grows by an extra fanout[0]
        #    factor relative to the standard estimate."
        if disjoint:
            amplification = effective_fanout[0]
            _dbg_disjoint(
                "DisjointMemoryEstimator.calc_seeds_per_call",
                f"disjoint=True → fanout_prod {fanout_prod} × fanout[0]={amplification} "
                f"= {fanout_prod * amplification}"
            )
            fanout_prod *= amplification

        result = int(cls.BASE_VERTICES_PER_BYTE * total_memory_bytes / fanout_prod)

        _dbg_disjoint(
            "DisjointMemoryEstimator.calc_seeds_per_call",
            f"effective_fanout={effective_fanout} fanout_prod={fanout_prod} "
            f"disjoint={disjoint} result={result} seeds_per_call"
        )
        return result


# ─── WalpurgisDisjointSession: 对应 NeighborLoader/LinkNeighborLoader 的 disjoint 路径 ──
# Python (b25bc88):
#   neighbor_loader.py:
#     删除: if disjoint: raise ValueError("Disjoint sampling is currently unsupported")
#     新增: disjoint=disjoint 传入 DistributedNeighborSampler
#   link_neighbor_loader.py: 同上
#
# 改写: 封装为 WalpurgisDisjointSession, 携带 disjoint 配置 + 内存估算
#       Python 是 bool flag 透传, 我们改写为可审计的配置对象
class WalpurgisDisjointSession:
    """
    Encapsulates disjoint sampling session for one NeighborLoader/LinkNeighborLoader call.

    b25bc88 对 Python 的改变:
      - NeighborLoader/LinkNeighborLoader: 删除 disjoint ValueError, 传 disjoint=disjoint
      - DistributedNeighborSampler: 接收 disjoint, 写入 sample_kwargs + calc_seeds

    改写比 Python 更结构化:
      - DisjointSamplingConfig 封装 sample_kwargs 构建
      - DisjointMemoryEstimator 封装 seeds_per_call 估算 (可单独测试)
      - validate() 方法: 检查 disjoint + fanout 兼容性
    """

    def __init__(
        self,
        disjoint: bool = False,
        fanout: Optional[List[int]] = None,
        total_memory_bytes: int = 0,
        heterogeneous: bool = False,
        num_edge_types: int = 1,
        compression: str = "COO",
        compress_per_hop: bool = False,
        with_replacement: bool = False,
        local_seeds_per_call: Optional[int] = None,
    ):
        self.disjoint = disjoint
        self.fanout = fanout or []
        self.heterogeneous = heterogeneous
        self.num_edge_types = num_edge_types

        # 构建 DisjointSamplingConfig
        self.config = DisjointSamplingConfig(
            disjoint=disjoint,
            compression=compression,
            compress_per_hop=compress_per_hop,
            with_replacement=with_replacement,
        )

        # 估算 local_seeds_per_call (b25bc88 __calc_local_seeds_per_call)
        if total_memory_bytes > 0 and self.fanout:
            self.estimated_seeds_per_call = DisjointMemoryEstimator.calc_seeds_per_call(
                fanout=self.fanout,
                total_memory_bytes=total_memory_bytes,
                local_seeds_per_call=local_seeds_per_call,
                heterogeneous=heterogeneous,
                disjoint=disjoint,
                num_edge_types=num_edge_types,
            )
        else:
            self.estimated_seeds_per_call = (
                local_seeds_per_call
                if local_seeds_per_call is not None
                else DisjointMemoryEstimator.UNKNOWN_VERTICES_DEFAULT
            )

        _dbg_disjoint(
            "WalpurgisDisjointSession.__init__",
            f"disjoint={disjoint} fanout={self.fanout} "
            f"hetero={heterogeneous} num_edge_types={num_edge_types} "
            f"estimated_seeds_per_call={self.estimated_seeds_per_call}"
        )

    def validate(self) -> bool:
        """
        Validate session consistency.
        断点调试: 打印完整配置摘要.

        b25bc88 没有新增 validation, 但上游删除了 ValueError,
        我们在此补充 soft validation (warning 而非 raise)
        """
        ok = True

        # disjoint + fanout[0]=0 → memory estimate 为 0 (除零风险)
        if self.disjoint and self.fanout and self.fanout[0] <= 0:
            print(
                f"[WARN b25bc88 WalpurgisDisjointSession.validate] "
                f"disjoint=True but fanout[0]={self.fanout[0]} <= 0 — "
                f"memory amplification undefined; falling back to UNKNOWN_VERTICES_DEFAULT",
                file=sys.stderr
            )
            ok = False

        # disjoint + hetero + fanout 长度不整除 → normalize 会 raise
        if self.disjoint and self.heterogeneous and self.fanout:
            if len(self.fanout) % self.num_edge_types != 0:
                print(
                    f"[WARN b25bc88 WalpurgisDisjointSession.validate] "
                    f"disjoint=True + heterogeneous=True but "
                    f"len(fanout)={len(self.fanout)} % num_edge_types={self.num_edge_types} != 0",
                    file=sys.stderr
                )
                ok = False

        _dbg_disjoint(
            "WalpurgisDisjointSession.validate",
            f"disjoint={self.disjoint} fanout={self.fanout} "
            f"estimated_seeds={self.estimated_seeds_per_call} valid={ok}"
        )
        return ok

    def dump_state(self) -> None:
        """断点调试: 打印 WalpurgisDisjointSession 完整状态"""
        print(
            f"[DEBUG b25bc88 WalpurgisDisjointSession.dump_state]\n"
            f"  disjoint={self.disjoint}\n"
            f"  fanout={self.fanout}\n"
            f"  heterogeneous={self.heterogeneous}  num_edge_types={self.num_edge_types}\n"
            f"  estimated_seeds_per_call={self.estimated_seeds_per_call}",
            file=sys.stderr
        )
        self.config.dump()


# ─── validate_disjoint_batches: 对应 test_neighbor_loader_disjoint_batch_structure ──
# Python (b25bc88 test_neighbor_loader.py:818-852):
#   for batch in loader:
#       tree_vertices = {n_id: set(...) for n_id in batch.input_id}
#       for hop in range(len(batch.num_sampled_edges)):
#           e_h = batch.edge_index[:, edge_offset:edge_offset+edges_hop]
#           tree_vertices[n_id].update(e_h[0][e_in].tolist())
#       for i, j: assert tv_items[i] & tv_items[j] == set()
#
# 改写: 提取为独立函数, 加断点调试 + overlap 统计
# (Python 是 test 内联逻辑, 改写为可复用验证器)
def validate_disjoint_batches(
    batches: List[Any],
    *,
    print_overlap_stats: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that per-seed subgraphs are disjoint across all batches.

    对应 b25bc88 test_neighbor_loader_disjoint_batch_structure 的验证逻辑.

    Python: test 内联, 仅 assert, 无统计
    改写: 提取为验证函数, 返回 (all_disjoint, stats), 加 overlap 统计

    Args:
        batches: List of batch objects with .input_id, .edge_index, .num_sampled_edges
        print_overlap_stats: 是否打印 overlap 统计 (默认 False, WALPURGIS_DEBUG=1 时自动打印)

    Returns:
        (all_disjoint: bool, stats: dict with overlap_count, total_pairs, etc.)
    """
    print_stats = print_overlap_stats or _DBG
    total_pairs = 0
    overlap_pairs = 0
    total_batches = 0

    all_disjoint = True

    for batch_idx, batch in enumerate(batches):
        total_batches += 1
        tree_vertices: Dict[Any, set] = {}

        # b25bc88 test 逻辑: per-seed 树遍历
        for n_id in batch.input_id:
            key = n_id.item() if hasattr(n_id, 'item') else n_id
            tree_vertices[key] = {key}

            edge_offset = 0
            for hop in range(len(batch.num_sampled_edges)):
                edges_hop = batch.num_sampled_edges[hop]
                if edges_hop == 0:
                    continue

                e_h = batch.edge_index[:, edge_offset: edge_offset + edges_hop]

                # 找属于当前 seed 子树的入边
                current_set = tree_vertices[key]
                # 改写: 不依赖 torch.isin, 用 Python set 操作 (可在 CPU 上运行)
                dst_nodes = e_h[1].tolist() if hasattr(e_h[1], 'tolist') else list(e_h[1])
                src_nodes = e_h[0].tolist() if hasattr(e_h[0], 'tolist') else list(e_h[0])

                for src, dst in zip(src_nodes, dst_nodes):
                    if dst in current_set:
                        tree_vertices[key].add(src)

                edge_offset += edges_hop

        # b25bc88 test: 检查所有 seed 对的子图互不相交
        tv_items = list(tree_vertices.values())
        for i in range(len(tv_items)):
            for j in range(i + 1, len(tv_items)):
                total_pairs += 1
                overlap = tv_items[i] & tv_items[j]
                if overlap:
                    overlap_pairs += 1
                    all_disjoint = False
                    _dbg_disjoint(
                        "validate_disjoint_batches",
                        f"batch_idx={batch_idx} seed_pair=({i},{j}) "
                        f"OVERLAP={overlap} (b25bc88 test 应为空集)"
                    )

    stats = {
        "all_disjoint": all_disjoint,
        "total_batches": total_batches,
        "total_pairs": total_pairs,
        "overlap_pairs": overlap_pairs,
        "overlap_rate": overlap_pairs / max(total_pairs, 1),
    }

    if print_stats:
        print(
            f"[DEBUG b25bc88 validate_disjoint_batches] "
            f"all_disjoint={all_disjoint} "
            f"batches={total_batches} pairs={total_pairs} "
            f"overlaps={overlap_pairs} rate={stats['overlap_rate']:.4f}",
            file=sys.stderr
        )

    return all_disjoint, stats


# ─── Convenience builder: 对应 NeighborLoader.__init__ disjoint 路径 ──────────
def make_disjoint_session(
    disjoint: bool,
    fanout: List[int],
    total_memory_bytes: int = 0,
    *,
    heterogeneous: bool = False,
    num_edge_types: int = 1,
    compression: str = "COO",
    with_replacement: bool = False,
    local_seeds_per_call: Optional[int] = None,
) -> WalpurgisDisjointSession:
    """
    建立 WalpurgisDisjointSession, 对应 b25bc88 NeighborLoader/LinkNeighborLoader
    __init__ 中从 loader args 到 DistributedNeighborSampler 的 disjoint 配置.

    Usage (mirrors b25bc88 neighbor_loader.py):
        session = make_disjoint_session(
            disjoint=True,
            fanout=[10, 5],
            total_memory_bytes=gpu_mem,
            heterogeneous=False,
        )
        assert session.config.disjoint is True
        sample_kwargs = session.config.to_sample_kwargs()
        assert sample_kwargs["disjoint_sampling"] is True

    断点调试: WALPURGIS_DEBUG=1 时打印完整 session 状态
    """
    _dbg_disjoint(
        "make_disjoint_session",
        f"disjoint={disjoint} fanout={fanout} "
        f"total_memory_bytes={total_memory_bytes} "
        f"heterogeneous={heterogeneous} num_edge_types={num_edge_types}"
    )

    session = WalpurgisDisjointSession(
        disjoint=disjoint,
        fanout=fanout,
        total_memory_bytes=total_memory_bytes,
        heterogeneous=heterogeneous,
        num_edge_types=num_edge_types,
        compression=compression,
        compress_per_hop=False,
        with_replacement=with_replacement,
        local_seeds_per_call=local_seeds_per_call,
    )

    if _DBG:
        session.dump_state()
        session.validate()

    return session
