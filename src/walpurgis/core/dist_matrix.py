"""
dist_matrix.py — d43e6c1 迁移: 修警告、修 MNMG graph store 测试、修 Matrix Accessors

migrate d43e6c1: [BUG] Fix warnings, fix MNMG graph store test, Matrix Accessors

上游变化 (d43e6c1, cugraph-gnn / python/cugraph-pyg/):

1. tensor/dist_matrix.py — Matrix Accessor 彻底重写 (核心 bug 修复):
   旧实现:
     local_col: return self._col.get_local_tensor()
     local_row: return self._row.get_local_tensor()
     local_coo: return torch.stack(self.get_local_tensor())
   新实现:
     local_col / local_row: 按 rank 手动切片，计算公式:
       q = sz // world_size   ← 整除基础块大小
       r = sz % world_size    ← 余数（前 r 个 rank 多分 1 条边）
       if rank < r:
           ix = arange(q*rank + rank, q*(rank+1) + rank + 1)   # 长 q+1
       else:
           ix = arange(q*rank + r, q*(rank+1) + r)             # 长 q
       return self._col[ix]
   local_coo: torch.stack([self.local_col, self.local_row])
              (原先错误传入 self.get_local_tensor() 而非 local_col/local_row)

2. data/graph_store.py — 三处 int() 类型转换 + 一处键名 bug:
   a. num_vertices 赋值时增加 int() 强制转换，防止 numpy/tensor 标量进入 dict
      导致后续比较或序列化异常
   b. edge_type[1] → edge_type[2] (关键 key 错误):
      原代码在 edge_type[0] == edge_type[2] (同构图) 分支中，
      num_vertices[edge_type[1]] = int(...) 写入了关系名而非 dst 顶点类型，
      导致 num_vertices 字典键乱套，后续图构建读取顶点数时全部 KeyError

3. tests/data/test_graph_store_mg.py — MNMG 测试修复:
   a. src/dst 强转 .to(torch.int64) — 避免 int32 vs int64 类型不匹配的 NCCL 报错
   b. 分布式验证逻辑从直接比较局部 rei 改为 all_gather 后全局对比:
      旧: assert (local_dst == rei[0]).all()  ← 每 rank 只验证自己那片，漏掉跨 rank 乱序问题
      新: all_gather sizes → 按大小 all_gather rei → concat → assert (gathered == expected).all()

Bug 根因 (Knuth 审查):

1. diff 对比源:
   | 上游原始          | d43e6c1 修复后        | Walpurgis 迁移         |
   |---|---|---|
   | get_local_tensor() 黑盒  | 手工分片 ix 计算      | SlicePartitioner 策略对象 |
   | local_coo 传旧 API       | [local_col, local_row]| 复用 SlicePartitioner    |
   | num_vertices 无 int()    | int(max(...))         | VertexCountRegistry 断言 |
   | edge_type[1] 关系名键    | edge_type[2] 正确 dst | 注释标注为 "silent typo" |
   | 局部 rei 直接比较        | all_gather 全局比较   | GatheredEdgeVerifier     |

2. 用户角度 bug:
   - 最严重：edge_type[1] 是关系名（如 "knows"），edge_type[2] 才是 dst 顶点类型（如 "person"）。
     写错键后 num_vertices["knows"] = 34，而后续 _graph 构建时查 num_vertices["person"] → KeyError。
     整个 MNMG 图构建静默错误，仅在 SGGraph/MGGraph 构建时爆出莫名 KeyError。
   - num_vertices 若存入 numpy.int64/torch.Tensor 标量，在某些 Python 版本下
     dict key 类型混用（int vs numpy.int64）导致 max() 比较抛 TypeError（numpy 2.x）。
   - 旧测试局部比较：rank-0 验证自己的 32 条边正确，rank-1 同理，但若 all_gather 后
     全局边集合有重复/乱序，两 rank 均不知情。新测试 all_gather 后全局 assert 堵住此漏洞。

3. 系统角度安全:
   - get_local_tensor() 是 WholeGraph 的内部 API，无公开契约，任何版本升级都可能改变
     其返回格式（是否含 padding、是否含哨兵行等）。手工按 rank 切片依赖数学公式，
     语义明确，与 WholeGraph 内部实现解耦，是更安全的长期做法。
   - int() 转换防止 numpy 标量泄漏进 Python dict，在 numpy 2.x 中 numpy.int64 的
     __eq__ 行为已改变，导致 dict 比较逻辑不一致。强转 int 是防御性编程。
   - 版权年份 2025 → 2025-2026 是合规变更，说明文件在 2026 年有实质修改。

Walpurgis 改写 20% (鲁迅拿法):
- SlicePartitioner: 替代 local_col/local_row 里重复出现的切片公式，
  提取为可单元测试的策略对象，携带 (rank, world_size, sz) → ix 映射
- VertexCountRegistry: 替代 num_vertices dict 的裸赋值，加 int() 断言 + debug print，
  put() 接口确保类型安全，与 UnifiedStoreRegistry 同族
- GatheredEdgeVerifier: 封装 all_gather sizes → all_gather tensors → concat → assert 流程，
  在测试文件中替代重复的分布式聚合样板代码
- 全链路 WALPURGIS_DEBUG=1 断点 print

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
from typing import Optional, Tuple

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS dist_matrix]", *args, file=sys.stderr, flush=True, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# SlicePartitioner — 替代 local_col / local_row 里重复的切片公式
# ──────────────────────────────────────────────────────────────────────────────

class SlicePartitioner:
    """
    封装 d43e6c1 引入的按 rank 均匀分片逻辑。

    上游新实现 (dist_matrix.py local_col / local_row):
        q = sz // world_size
        r = sz % world_size
        if rank < r:
            ix = torch.arange(q * rank + rank, q * (rank + 1) + rank + 1)
        else:
            ix = torch.arange(q * rank + r, q * (rank + 1) + r)

    数学语义:
        - 总 sz 条边均匀分给 world_size 个 rank
        - 前 r 个 rank 每人多分 1 条（长度 q+1）
        - 后 world_size-r 个 rank 每人分 q 条（长度 q）
        - rank < r 时起点 = q*rank + rank（含前面所有多分的 1）
        - rank >= r 时起点 = q*rank + r（前 r 个 rank 各多了 1）

    公式验证:
        rank=0, r>0: ix = [0, q+1)        长度 q+1 ✓
        rank=1, r>1: ix = [q+1, 2q+2)     长度 q+1 ✓
        rank=r,      ix = [q*r+r, q*(r+1)+r) = [r*(q+1), r*(q+1)+q)  长度 q ✓

    Walpurgis 改写:
    - compute_indices() 静态方法：给定 (sz, rank, world_size) 返回 ix tensor
    - slice_tensor() 静态方法：compute_indices + tensor[ix]，一步到位
    - WALPURGIS_DEBUG=1 时打印分片决策

    断点 1: compute_indices() 入口，打印 sz, rank, world_size, q, r
    断点 2: compute_indices() 完成，打印 ix.shape, ix[0], ix[-1]
    """

    @staticmethod
    def compute_indices(sz: int, rank: int, world_size: int):
        """
        计算当前 rank 应读取的下标范围 ix。

        参数
        ----
        sz         : tensor 总长度
        rank       : 当前进程 rank
        world_size : 总进程数

        返回
        ----
        torch.Tensor: 1D int64 index tensor
        """
        import torch

        q = sz // world_size
        r = sz % world_size

        # ── 断点 1: 进入 compute_indices ──────────────────────────────────
        _dbg(
            f"SlicePartitioner.compute_indices(): "
            f"sz={sz}, rank={rank}, world_size={world_size}, "
            f"q={q}, r={r}"
        )

        if rank < r:
            # 前 r 个 rank，各自多分 1 条边
            start = q * rank + rank
            end = q * (rank + 1) + rank + 1
        else:
            # 后 world_size-r 个 rank，标准 q 条边
            start = q * rank + r
            end = q * (rank + 1) + r

        ix = torch.arange(start, end, dtype=torch.int64)

        # ── 断点 2: compute_indices 完成 ──────────────────────────────────
        _dbg(
            f"SlicePartitioner.compute_indices() → "
            f"ix.shape={tuple(ix.shape)}, "
            f"start={start}, end={end}, "
            f"len={end - start}"
        )

        return ix

    @staticmethod
    def slice_tensor(tensor, rank: int, world_size: int):
        """
        按 rank 切片 tensor，返回当前 rank 对应的局部子张量。

        参数
        ----
        tensor     : 1D torch.Tensor，全局边索引
        rank       : 当前进程 rank
        world_size : 总进程数

        返回
        ----
        torch.Tensor: 局部切片

        断点 3: 打印切片前后 shape
        """
        sz = tensor.shape[0]
        ix = SlicePartitioner.compute_indices(sz, rank, world_size)

        # ── 断点 3 ────────────────────────────────────────────────────────
        _dbg(
            f"SlicePartitioner.slice_tensor(): "
            f"tensor.shape={tuple(tensor.shape)}, "
            f"ix.numel()={ix.numel()}"
        )

        result = tensor[ix]

        _dbg(f"  → local slice shape={tuple(result.shape)}")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# VertexCountRegistry — 替代 num_vertices dict 裸赋值
# ──────────────────────────────────────────────────────────────────────────────

class VertexCountRegistry:
    """
    封装 d43e6c1 graph_store.py 中 num_vertices dict 的赋值逻辑。

    上游 d43e6c1 修复了两类问题:
        A. int() 强制转换：
           - 原: num_vertices[t] = max(num_vertices[t], edge_attr.size[0])
           - 新: num_vertices[t] = int(max(...))
           防止 numpy.int64 / torch.Tensor 标量污染 dict，
           numpy 2.x 中 numpy.int64 与 Python int 的 __hash__ / __eq__ 行为改变，
           导致 dict 键查找不一致。

        B. edge_type 键名 bug (silent typo):
           - 原: num_vertices[edge_attr.edge_type[1]] = int(...)
                 edge_type = (src_type, rel_type, dst_type)
                 edge_type[1] = rel_type（关系名，如 "knows"）← 错误！
           - 新: num_vertices[edge_attr.edge_type[2]] = int(...)
                 edge_type[2] = dst_type（目标顶点类型，如 "person"）← 正确

    三元组约定（PyG/cuGraph 统一）:
        edge_type = (src_type, rel_type, dst_type)
        edge_type[0] = src_type   ← 源顶点类型
        edge_type[1] = rel_type   ← 关系名（字符串，不能作为顶点数量 key！）
        edge_type[2] = dst_type   ← 目标顶点类型

    Walpurgis 改写:
    - update_from_size(): 替代 if t in num_vertices ... else ... 三元表达式，
      加 int() 断言 + debug print
    - update_from_index(): 替代 num_vertices[edge_type[2]] = int(max()+1) 模式
    - get_dst_type_key(): 明确提取 edge_type[2]，防止再次写成 [1]
    - WALPURGIS_DEBUG=1 时打印每次 put 前后的值变化

    断点 4: update_from_size() 入口，打印类型名 + 旧值 + 新 size
    断点 5: update_from_index() 入口，打印类型名 + 来源 tensor max
    断点 6: get_dst_type_key() — 明确打印 edge_type 三元组，避免 [1] vs [2] 混淆
    """

    def __init__(self):
        self._counts = {}  # Dict[str, int]

    def update_from_size(self, vertex_type: str, size_val) -> None:
        """
        用 edge_attr.size[i] 更新顶点数量，取 max（与 d43e6c1 语义一致）。

        上游对应代码:
            num_vertices[t] = int(max(num_vertices[t], size)) if t in num_vertices else int(size)

        参数
        ----
        vertex_type : str  顶点类型名（edge_type[0] 或 edge_type[2]，绝非 edge_type[1]！）
        size_val    : 新的顶点数量候选值（可能是 numpy.int64 / torch.Tensor 标量）
        """
        new_val = int(size_val)  # 防御性 int() 转换，与 d43e6c1 一致
        old_val = self._counts.get(vertex_type, None)

        # ── 断点 4 ────────────────────────────────────────────────────────
        _dbg(
            f"VertexCountRegistry.update_from_size(): "
            f"vertex_type={vertex_type!r}, "
            f"old={old_val}, size_val={size_val!r} → new_val={new_val}"
        )

        if old_val is not None:
            updated = int(max(old_val, new_val))
        else:
            updated = new_val

        _dbg(f"  → stored={updated} (was {old_val})")
        self._counts[vertex_type] = updated

    def update_from_index(self, vertex_type: str, index_tensor) -> None:
        """
        用边索引 tensor 的 max()+1 推断顶点数量。

        上游对应代码 (graph_store.py ~L264-L267):
            if edge_attr.edge_type[2] not in num_vertices:
                num_vertices[edge_attr.edge_type[2]] = int(
                    self.__edge_indices[edge_attr.edge_type].local_row.max() + 1
                )

        参数
        ----
        vertex_type  : str  顶点类型名（必须是 edge_type[2]，由调用方保证）
        index_tensor : torch.Tensor  边索引 tensor（local_row 或 local_col）

        断点 5
        """
        if vertex_type in self._counts:
            # 上游只在 not in 时更新，此处保持一致
            _dbg(
                f"VertexCountRegistry.update_from_index(): "
                f"vertex_type={vertex_type!r} 已存在 ({self._counts[vertex_type]})，跳过"
            )
            return

        derived = int(index_tensor.max() + 1)

        # ── 断点 5 ────────────────────────────────────────────────────────
        _dbg(
            f"VertexCountRegistry.update_from_index(): "
            f"vertex_type={vertex_type!r}, "
            f"index_tensor.max()={index_tensor.max().item()}, "
            f"derived={derived}"
        )

        self._counts[vertex_type] = derived

    @staticmethod
    def get_dst_type_key(edge_type: tuple) -> str:
        """
        安全提取 edge_type 三元组中的目标顶点类型（edge_type[2]）。

        d43e6c1 修复的 silent typo: 原代码写了 edge_type[1]（关系名）
        而非 edge_type[2]（目标顶点类型）。此方法明确提取 [2]，
        消除未来再次混淆 [1] vs [2] 的可能。

        断点 6: 打印 edge_type 三元组 + 所取下标
        """
        assert len(edge_type) == 3, f"edge_type 必须是三元组 (src, rel, dst)，got {edge_type}"
        src_type, rel_type, dst_type = edge_type

        # ── 断点 6 ────────────────────────────────────────────────────────
        _dbg(
            f"VertexCountRegistry.get_dst_type_key(): "
            f"edge_type=({src_type!r}, {rel_type!r}, {dst_type!r}) "
            f"→ dst_type={dst_type!r} [index=2, NOT 1]"
        )

        return dst_type

    def get(self, vertex_type: str, default=None):
        return self._counts.get(vertex_type, default)

    def __contains__(self, vertex_type):
        return vertex_type in self._counts

    def __getitem__(self, vertex_type):
        return self._counts[vertex_type]

    def items(self):
        return self._counts.items()

    def as_dict(self) -> dict:
        """返回与上游 num_vertices dict 等价的纯 Python int dict。"""
        return dict(self._counts)

    def __repr__(self):
        return f"VertexCountRegistry({self._counts})"


# ──────────────────────────────────────────────────────────────────────────────
# GatheredEdgeVerifier — 封装 test_graph_store_mg.py 的 all_gather 验证逻辑
# ──────────────────────────────────────────────────────────────────────────────

class GatheredEdgeVerifier:
    """
    封装 d43e6c1 test_graph_store_mg.py 中新增的分布式边集合验证逻辑。

    d43e6c1 测试修复 (test_graph_store_mg.py):
        旧验证（局部对比，有盲区）:
            assert (local_dst == rei[0]).all()
            assert (local_src == rei[1]).all()
            ← 每 rank 只验证自己那片，跨 rank 乱序/重复不可知

        新验证（全局 all_gather 后对比）:
            # Step 1: 各 rank gather 自己的 rei.shape[1]
            local_size = torch.tensor([rei.shape[1]], device=f"cuda:{rank}")
            gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_sizes, local_size)
            # Step 2: 按 size 分配 buffer，all_gather rei
            gathered_rei = [torch.zeros((2, size.item()), dtype=rei.dtype, ...) for size in gathered_sizes]
            torch.distributed.all_gather(gathered_rei, rei)
            gathered_rei = torch.concat(gathered_rei, dim=1)
            # Step 3: 全局断言
            assert (gathered_rei == torch.stack([dst, src])).all()

    分布式验证的必要性:
        - 每 rank 分到的 rei 是全局 edgelist 的一个切片，局部正确不代表全局正确
        - all_gather 后的 concat 才能与完整的 [dst, src] 期望值比对
        - 若 num_vertices 的 edge_type[1] bug 导致图构建时 src/dst 顶点偏移错误，
          全局 assert 一定会失败，而局部 assert 可能侥幸通过

    Walpurgis 改写:
    - gather_edge_index(): 封装 sizes all_gather + rei all_gather + concat 三步
    - verify_against(): 全局 concat 后 assert，打印详细错误信息
    - WALPURGIS_DEBUG=1 时打印 per-rank sizes + concat 后 shape

    断点 7: gather_edge_index() 入口，打印 local rei shape
    断点 8: gather_edge_index() 完成，打印 gathered sizes + concat shape
    断点 9: verify_against() 断言失败时打印 diff 摘要
    """

    @staticmethod
    def gather_edge_index(rei, rank: int, world_size: int, device: str):
        """
        在所有 rank 上 all_gather rei，返回全局拼接后的 edge index。

        参数
        ----
        rei        : torch.Tensor, shape=(2, local_edges)，本 rank 的 edge index
        rank       : 当前 rank
        world_size : 总 rank 数
        device     : 如 "cuda:0"

        返回
        ----
        torch.Tensor: shape=(2, total_edges)，所有 rank 的边拼接
        """
        import torch

        # ── 断点 7 ────────────────────────────────────────────────────────
        _dbg(
            f"GatheredEdgeVerifier.gather_edge_index(): "
            f"rank={rank}, world_size={world_size}, "
            f"local rei.shape={tuple(rei.shape)}"
        )

        # Step 1: 收集各 rank 的边数
        local_size = torch.tensor([rei.shape[1]], device=device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_size)

        gathered_size_vals = [int(s.item()) for s in gathered_sizes]
        _dbg(f"  gathered_sizes per rank: {gathered_size_vals}")

        # Step 2: 按各 rank 实际大小分配 buffer，all_gather rei
        gathered_rei = [
            torch.zeros((2, size), dtype=rei.dtype, device=device)
            for size in gathered_size_vals
        ]
        torch.distributed.all_gather(gathered_rei, rei)

        # Step 3: 拼接成全局 edge index
        result = torch.concat(gathered_rei, dim=1)

        # ── 断点 8 ────────────────────────────────────────────────────────
        _dbg(
            f"GatheredEdgeVerifier.gather_edge_index() 完成: "
            f"concat shape={tuple(result.shape)}, "
            f"total_edges={result.shape[1]}"
        )

        return result

    @staticmethod
    def verify_against(gathered_rei, expected_dst, expected_src):
        """
        对比 all_gather 后的全局 rei 与期望的 [dst, src]。

        上游断言:
            assert (gathered_rei == torch.stack([dst, src])).all()

        参数
        ----
        gathered_rei  : shape=(2, total_edges)，all_gather 后拼接
        expected_dst  : shape=(total_edges,)，期望 dst（PyG row）
        expected_src  : shape=(total_edges,)，期望 src（PyG col）

        断点 9: 断言失败时打印不匹配的前 5 条
        """
        import torch

        expected = torch.stack([expected_dst, expected_src])

        # ── 断点 9: 验证逻辑 ──────────────────────────────────────────────
        _dbg(
            f"GatheredEdgeVerifier.verify_against(): "
            f"gathered_rei.shape={tuple(gathered_rei.shape)}, "
            f"expected.shape={tuple(expected.shape)}"
        )

        match = (gathered_rei == expected).all()

        if not match:
            # 找出第一个不匹配位置，打印诊断信息
            diff_mask = (gathered_rei != expected)
            diff_cols = diff_mask.any(dim=0).nonzero().squeeze()
            first_few = diff_cols[:5] if diff_cols.numel() > 5 else diff_cols
            _dbg(
                f"  !! 验证失败: {diff_mask.sum().item()} 处不匹配，"
                f"前几列 indices={first_few.tolist()}"
            )
            _dbg(f"  gathered_rei[:, :5]={gathered_rei[:, :5].tolist()}")
            _dbg(f"  expected[:, :5]={expected[:, :5].tolist()}")

        assert match, (
            f"GatheredEdgeVerifier: edge index 不匹配！"
            f"gathered shape={tuple(gathered_rei.shape)}, "
            f"expected shape={tuple(expected.shape)}"
        )

        _dbg("  ✓ 全局 edge index 验证通过")


# ──────────────────────────────────────────────────────────────────────────────
# WalpurgisDistMatrix — d43e6c1 dist_matrix.py 的 Walpurgis 封装版本
# ──────────────────────────────────────────────────────────────────────────────

class WalpurgisDistMatrix:
    """
    d43e6c1 DistMatrix 的 Walpurgis 迁移封装。

    上游 DistMatrix (cugraph_pyg/tensor/dist_matrix.py):
        - _col / _row: 两个 DistTensor，存储 COO 格式边索引的列/行
        - local_col: 当前 rank 对应的 _col 切片
        - local_row: 当前 rank 对应的 _row 切片
        - local_coo: torch.stack([local_col, local_row])
        - shape: (num_dst_vertices, num_src_vertices)

    d43e6c1 的核心修复:
        local_col / local_row 从调用 get_local_tensor()（WholeGraph 内部 API，
        语义不稳定）改为手工按 rank 切片，与 SlicePartitioner 一致。
        local_coo 从错误的 torch.stack(self.get_local_tensor()) 改为
        torch.stack([self.local_col, self.local_row])。

    Walpurgis 改写:
    - 使用 SlicePartitioner 替代内联切片公式，消除代码重复
    - WALPURGIS_DEBUG=1 时打印 local_col / local_row shape

    断点 10: local_col property 被调用，打印 rank + sz + ix range
    断点 11: local_row property 被调用，打印 rank + sz + ix range
    断点 12: local_coo property 被调用，打印 stack 后 shape
    """

    def __init__(self, col_tensor, row_tensor):
        """
        参数
        ----
        col_tensor : DistTensor 或任何支持 tensor[ix] 的对象，存储 COO 列（dst）
        row_tensor : DistTensor 或任何支持 tensor[ix] 的对象，存储 COO 行（src）
        """
        self._col = col_tensor
        self._row = row_tensor

    @property
    def local_col(self):
        """
        返回当前 rank 对应的 _col（dst 边索引）切片。

        d43e6c1 新实现：手工计算 ix，替代 get_local_tensor()
        Walpurgis：委托 SlicePartitioner.slice_tensor()

        断点 10
        """
        import torch

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sz = self._col.shape[0]

        # ── 断点 10 ───────────────────────────────────────────────────────
        _dbg(
            f"WalpurgisDistMatrix.local_col: "
            f"rank={rank}, world_size={world_size}, sz={sz}"
        )

        result = SlicePartitioner.slice_tensor(self._col, rank, world_size)
        _dbg(f"  → local_col.shape={tuple(result.shape)}")
        return result

    @property
    def local_row(self):
        """
        返回当前 rank 对应的 _row（src 边索引）切片。

        断点 11
        """
        import torch

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sz = self._row.shape[0]

        # ── 断点 11 ───────────────────────────────────────────────────────
        _dbg(
            f"WalpurgisDistMatrix.local_row: "
            f"rank={rank}, world_size={world_size}, sz={sz}"
        )

        result = SlicePartitioner.slice_tensor(self._row, rank, world_size)
        _dbg(f"  → local_row.shape={tuple(result.shape)}")
        return result

    @property
    def local_coo(self):
        """
        返回 [local_col, local_row] 堆叠后的 COO tensor，shape=(2, local_edges)。

        d43e6c1 修复: 原来错误地传入 self.get_local_tensor() (已删除的旧 API)，
        现在正确地使用 [self.local_col, self.local_row]。

        断点 12
        """
        import torch

        col = self.local_col
        row = self.local_row

        # ── 断点 12 ───────────────────────────────────────────────────────
        _dbg(
            f"WalpurgisDistMatrix.local_coo: "
            f"col.shape={tuple(col.shape)}, row.shape={tuple(row.shape)}"
        )

        result = torch.stack([col, row])
        _dbg(f"  → local_coo.shape={tuple(result.shape)}")
        return result

    @property
    def shape(self):
        """返回 (col 总长, row 总长)，与上游 DistMatrix.shape 一致。"""
        return (self._col.shape[0], self._row.shape[0])

    def __repr__(self):
        return (
            f"WalpurgisDistMatrix("
            f"col.shape={self._col.shape}, "
            f"row.shape={self._row.shape})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 使用示例（WALPURGIS_DEBUG=1 下可验证每个断点）
# ──────────────────────────────────────────────────────────────────────────────
#
# # SlicePartitioner 单机测试（无 torch.distributed）
# import torch
# col_tensor = torch.arange(34)   # 34 条边
# ix = SlicePartitioner.compute_indices(sz=34, rank=0, world_size=2)
# print(ix)   # tensor([ 0,  1, ..., 16])  长度 17
# ix1 = SlicePartitioner.compute_indices(sz=34, rank=1, world_size=2)
# print(ix1)  # tensor([17, 18, ..., 33])  长度 17
#
# # VertexCountRegistry 键名安全性验证
# reg = VertexCountRegistry()
# edge_type = ("person", "knows", "person")
# dst_key = VertexCountRegistry.get_dst_type_key(edge_type)  # "person"（index=2）
# reg.update_from_size(dst_key, 34)
# print(reg)  # VertexCountRegistry({'person': 34})
#
# # GatheredEdgeVerifier 使用（需 torch.distributed 环境）
# # gathered = GatheredEdgeVerifier.gather_edge_index(rei, rank=0, world_size=2, device="cuda:0")
# # GatheredEdgeVerifier.verify_against(gathered, dst, src)
