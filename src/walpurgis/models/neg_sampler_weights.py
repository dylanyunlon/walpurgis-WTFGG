"""
neg_sampler_weights.py — 7ea1138 迁移: 负采样权重拼接逻辑修复

migrate 7ea1138: [BUG] Fix Weights Issue in Negative Sampling

上游变化 (7ea1138, cugraph-gnn, alexbarghi-nv, 2026-04-08):
  文件: python/cugraph-pyg/cugraph_pyg/sampler/sampler_utils.py
  函数: neg_sample()

  bug: src_weight / dst_weight 的补零 concat 在所有异构图路径无条件执行，
       包括 src==dst（同类型节点，如 author→author）的情况。
       而 src==dst 时 src_weight 已是 num_src_nodes 长度（填满），
       再 concat zeros(num_dst_nodes) 会导致权重向量长度翻倍，
       远超 vertices 长度，pylibcugraph 负采样引擎按偏移索引进入越界区域，
       采样结果静默错误（采到不存在节点ID）。

  fix: 将 src_weight / dst_weight concat 移入 `if input_type[0] != input_type[2]:`
       分支——仅在 src≠dst 类型（需要拼接两半权重）时执行。
       src==dst 分支（else）只设置 vertices，不修改已经完整的权重数组。
       同时删除 `elif src_weight is None and dst_weight is None: vertices = None`
       分支——在此之前权重已被填充为 ones/zeros，此条件永远不会为真，
       是死代码（原有逻辑缺陷的遗留）。

  diff 精读 (逐行):
    - 删除: `else:` 后的 vertices 赋值处于 outdent 层（if input_type...之外）
      → 原先 else 仅处理 vertices，但 src/dst_weight concat 在 else 外执行
    + 新增: src_weight concat 移入 `if input_type[0] != input_type[2]:` 块内
    + 新增: dst_weight concat 同样移入同一块内
    - 删除: 旧的块外 src_weight = torch.concat([...]) 两行
    - 删除: elif src_weight is None and dst_weight is None: vertices = None
      → 此分支为死代码，src_weight/dst_weight 此时已被 ones 填充
    = 保留: else: vertices = arange(num_src_nodes) （均匀同构路径）

Walpurgis 改写20%（鲁迅拿法）:
  - Python 原文是单函数内 if/else 散落逻辑；
    改写为 NegSampleWeightPlan 值对象（携带 vertices + aligned src/dst weights）
    + NegSampleWeightBuilder.build() 工厂方法（封装全部分支决策）
  - Python 直接操作 tensor；改写：builder 返回 plan，plan.validate() 独立可测
  - Python 的补零 concat 是 inline torch.zeros；
    改写：WeightAligner._pad_src / _pad_dst 静态方法，对称命名，语义显式
  - Python 无 dead-code 注释；改写：_is_dead_branch() 方法文档化"永远不可达"路径
  - 断点调试: WALPURGIS_DEBUG=1 全链路 print，覆盖:
    1. build() 入口: hetero/homo, src_type==dst_type 判断
    2. src!=dst 分支: concat 前后 weight shape
    3. src==dst 分支: 仅 vertices，weight 不变
    4. 同构分支: vertices arange
    5. validate(): shape 一致性检查结果

作者: dylanyunlon<dogechat@163.com>
"""
import sys
import os
from dataclasses import dataclass
from typing import Optional, Any

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg(tag: str, msg: str) -> None:
    """断点调试: neg_sampler_weights 专用 print"""
    if _DBG:
        print(f"[DEBUG 7ea1138 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── NegSampleWeightPlan: 对应 neg_sample() 中的 vertices + src/dst weight ──
# Python 原文: vertices, src_weight, dst_weight 三个局部变量在 if/else 中赋值后
#             直接传给 _call_plc_negative_sampling(...)
# 改写: 封装为 Plan 对象，携带构建后的三元状态，可独立 validate()
@dataclass
class NegSampleWeightPlan:
    """
    构建完毕的负采样权重方案。

    对应 neg_sample() 中:
        _call_plc_negative_sampling(graph_store, num_neg, vertices, src_weight, dst_weight)

    改写: 将三个参数封装为 Plan，validate() 检查 shape 一致性（Python 原文无此检查）。

    7ea1138 bug: src==dst 时 weight 被错误扩展
        before-fix: src_weight.shape = [num_src + num_dst]  ← 多了 num_dst 个零
        after-fix:  src_weight.shape = [num_src]            ← 正确
    """
    vertices: Any                  # torch.Tensor | None
    src_weight: Any                # torch.Tensor（已 aligned）
    dst_weight: Any                # torch.Tensor（已 aligned）
    src_dst_same_type: bool        # input_type[0] == input_type[2]
    is_homogeneous: bool           # graph_store.is_homogeneous
    num_src_nodes: int
    num_dst_nodes: int

    def validate(self) -> bool:
        """
        断点调试: 检查 weight shape 与 vertices shape 一致性。
        7ea1138 修复的核心不变量:
          - src!=dst: len(vertices) == num_src + num_dst
                      len(src_weight) == num_src + num_dst
                      len(dst_weight) == num_src + num_dst
          - src==dst: len(vertices) == num_src
                      len(src_weight) == num_src  ← 7ea1138 fix
                      len(dst_weight) == num_src  ← 7ea1138 fix
          - homo:     vertices == arange(num_src)
                      len(src_weight) == num_src
                      len(dst_weight) == num_dst
        """
        def _len(t) -> int:
            if t is None:
                return -1
            if hasattr(t, 'numel'):   # torch.Tensor
                return t.numel()
            if hasattr(t, '__len__'):
                return len(t)
            return -1

        v_len   = _len(self.vertices)
        sw_len  = _len(self.src_weight)
        dw_len  = _len(self.dst_weight)

        _dbg("NegSampleWeightPlan.validate",
             f"vertices={v_len} src_weight={sw_len} dst_weight={dw_len} "
             f"src_dst_same={self.src_dst_same_type} homo={self.is_homogeneous} "
             f"num_src={self.num_src_nodes} num_dst={self.num_dst_nodes}")

        if self.is_homogeneous:
            # 同构: vertices = arange(num_src), weights 各自 num_src/num_dst
            ok = (v_len == self.num_src_nodes and
                  sw_len == self.num_src_nodes and
                  dw_len == self.num_dst_nodes)
            if not ok:
                print(
                    f"[ERROR 7ea1138 validate] homogeneous shape mismatch: "
                    f"vertices={v_len} (expected {self.num_src_nodes}) "
                    f"src_weight={sw_len} (expected {self.num_src_nodes}) "
                    f"dst_weight={dw_len} (expected {self.num_dst_nodes})",
                    file=sys.stderr
                )
            return ok

        if not self.src_dst_same_type:
            # src≠dst: vertices concat, weights 补零 concat（7ea1138 修复后的正确路径）
            expected = self.num_src_nodes + self.num_dst_nodes
            ok = (v_len == expected and sw_len == expected and dw_len == expected)
            if not ok:
                print(
                    f"[ERROR 7ea1138 validate] hetero src!=dst shape mismatch: "
                    f"expected={expected} "
                    f"vertices={v_len} src_weight={sw_len} dst_weight={dw_len} "
                    f"—— 若 src_weight={expected} instead of {self.num_src_nodes}, "
                    f"则为 7ea1138 修复前的 bug（权重多 concat 了一次）",
                    file=sys.stderr
                )
            return ok
        else:
            # src==dst: vertices = offset(arange), weights 不变（7ea1138 核心修复）
            ok = (v_len == self.num_src_nodes and
                  sw_len == self.num_src_nodes and
                  dw_len == self.num_dst_nodes)
            if not ok:
                # 精确诊断 7ea1138 bug
                if sw_len == self.num_src_nodes + self.num_dst_nodes:
                    print(
                        f"[ERROR 7ea1138 validate] 检测到 7ea1138 修复前的 bug! "
                        f"src==dst 路径下 src_weight 被错误 concat: "
                        f"got {sw_len}, expected {self.num_src_nodes}. "
                        f"（src_weight 多拼了 {self.num_dst_nodes} 个零）",
                        file=sys.stderr
                    )
                else:
                    print(
                        f"[ERROR 7ea1138 validate] hetero src==dst shape mismatch: "
                        f"vertices={v_len} (expected {self.num_src_nodes}) "
                        f"src_weight={sw_len} (expected {self.num_src_nodes}) "
                        f"dst_weight={dw_len} (expected {self.num_dst_nodes})",
                        file=sys.stderr
                    )
            return ok


# ─── WeightAligner: 对应 7ea1138 内的 torch.concat 操作 ─────────────────────
# Python 原文 (buggy, 修复前):
#   src_weight = torch.concat([src_weight, torch.zeros(num_dst_nodes, ...)])
#   dst_weight = torch.concat([torch.zeros(num_src_nodes, ...), dst_weight])
# Python (修复后, 7ea1138):
#   同上两行，但仅在 input_type[0] != input_type[2] 块内执行
# 改写: 提取为静态方法，命名对称（_pad_src_for_dst / _pad_dst_for_src），语义显式
class WeightAligner:
    """
    对应 7ea1138 中 neg_sample() 内的权重补零 concat 操作。

    命名约定（改写）:
      _pad_src_for_dst: src_weight 尾部追加 num_dst 个零
                        → src 节点在 dst 节点空间的"无权重"占位
                        对应: src_weight = concat([src_weight, zeros(num_dst)])
      _pad_dst_for_src: dst_weight 头部追加 num_src 个零
                        → dst 节点在 src 节点空间的"无权重"占位
                        对应: dst_weight = concat([zeros(num_src), dst_weight])

    7ea1138 修复确保这两个方法仅在 src≠dst 时调用。
    """

    @staticmethod
    def _pad_src_for_dst(src_weight, num_dst_nodes: int, weight_dtype, device="cuda"):
        """
        对应 7ea1138 修复后:
            src_weight = torch.concat(
                [src_weight, torch.zeros(num_dst_nodes, dtype=weight_dtype, device="cuda")]
            )

        断点调试: 打印 concat 前后 shape
        """
        try:
            import torch
            _dbg("WeightAligner._pad_src_for_dst",
                 f"src_weight.shape={list(src_weight.shape)} "
                 f"padding={num_dst_nodes} zeros dtype={weight_dtype}")

            result = torch.concat([
                src_weight,
                torch.zeros(num_dst_nodes, dtype=weight_dtype, device=device)
            ])

            _dbg("WeightAligner._pad_src_for_dst",
                 f"result.shape={list(result.shape)} "
                 f"(src={src_weight.numel()} + dst_zeros={num_dst_nodes})")
            return result
        except ImportError:
            # torch 不可用时（单元测试环境），返回占位列表
            result = list(src_weight) + [0.0] * num_dst_nodes
            _dbg("WeightAligner._pad_src_for_dst",
                 f"[no-torch] src_len={len(src_weight)} -> result_len={len(result)}")
            return result

    @staticmethod
    def _pad_dst_for_src(dst_weight, num_src_nodes: int, weight_dtype, device="cuda"):
        """
        对应 7ea1138 修复后:
            dst_weight = torch.concat(
                [torch.zeros(num_src_nodes, dtype=weight_dtype, device="cuda"), dst_weight]
            )

        断点调试: 打印 concat 前后 shape
        """
        try:
            import torch
            _dbg("WeightAligner._pad_dst_for_src",
                 f"dst_weight.shape={list(dst_weight.shape)} "
                 f"padding={num_src_nodes} zeros prefix dtype={weight_dtype}")

            result = torch.concat([
                torch.zeros(num_src_nodes, dtype=weight_dtype, device=device),
                dst_weight
            ])

            _dbg("WeightAligner._pad_dst_for_src",
                 f"result.shape={list(result.shape)} "
                 f"(src_zeros={num_src_nodes} + dst={dst_weight.numel()})")
            return result
        except ImportError:
            result = [0.0] * num_src_nodes + list(dst_weight)
            _dbg("WeightAligner._pad_dst_for_src",
                 f"[no-torch] dst_len={len(dst_weight)} -> result_len={len(result)}")
            return result

    @staticmethod
    def _is_dead_branch(src_weight, dst_weight) -> bool:
        """
        对应 7ea1138 删除的死代码分支:
            elif src_weight is None and dst_weight is None:
                vertices = None

        7ea1138 修复说明: 此时 src_weight 和 dst_weight 已在
        neg_sample() 上半段被强制填充为 ones():
            if src_weight is None:
                src_weight = torch.ones(num_src_nodes, ...)
            if dst_weight is None:
                dst_weight = torch.ones(num_dst_nodes, ...)

        因此 `src_weight is None and dst_weight is None` 在到达这里时
        永远为 False。此分支是死代码，7ea1138 将其删除。

        断点调试: 若此方法返回 True，说明存在新的代码路径问题。
        """
        is_dead = (src_weight is None and dst_weight is None)
        if is_dead:
            print(
                "[ERROR 7ea1138 _is_dead_branch] 不应到达此分支! "
                "src_weight 和 dst_weight 应已在 neg_sample() 上半段被填充为 ones()。"
                "这是 7ea1138 修复前的死代码路径被意外激活，请检查调用栈。",
                file=sys.stderr
            )
        return is_dead


# ─── NegSampleWeightBuilder: 核心分支决策，对应 neg_sample() 的主 if/else 树 ─
# Python 原文 (buggy, 修复前):
#   if not graph_store.is_homogeneous:
#       if input_type[0] != input_type[2]:
#           vertices = concat([...src..., ...dst...])
#       else:
#           vertices = offset(arange(num_src))
#       src_weight = concat([src_weight, zeros(num_dst)])   ← BUG: 在 else 外!
#       dst_weight = concat([zeros(num_src), dst_weight])   ← BUG: 在 else 外!
#   elif src_weight is None and dst_weight is None:         ← 死代码
#       vertices = None
#   else:
#       vertices = arange(num_src)
#
# Python 原文 (7ea1138 修复后):
#   if not graph_store.is_homogeneous:
#       if input_type[0] != input_type[2]:
#           vertices = concat([...src..., ...dst...])
#           src_weight = concat([src_weight, zeros(num_dst)])   ← 移入此块
#           dst_weight = concat([zeros(num_src), dst_weight])   ← 移入此块
#       else:
#           vertices = offset(arange(num_src))
#           # weight 不动 ← 7ea1138 修复的核心
#   else:
#       vertices = arange(num_src)
#       # 删除了 elif 死代码分支
class NegSampleWeightBuilder:
    """
    封装 neg_sample() 中 7ea1138 修复的分支决策逻辑。

    Python 原文是单函数内 30 行 if/else，改写为 builder 模式:
      - 每个分支明确命名 (_build_hetero_src_ne_dst / _build_hetero_src_eq_dst / _build_homo)
      - build() 是顶层入口，对应 Python 的 `if not graph_store.is_homogeneous:` 树
    """

    @staticmethod
    def _build_hetero_src_ne_dst(
        src_weight, dst_weight,
        num_src_nodes: int, num_dst_nodes: int,
        weight_dtype,
        vertex_offsets,       # graph_store._vertex_offsets
        src_type: str,        # input_type[0]
        dst_type: str,        # input_type[2]
    ) -> NegSampleWeightPlan:
        """
        对应 7ea1138 修复后的 `if input_type[0] != input_type[2]:` 块:

            vertices = torch.concat([
                torch.arange(num_src_nodes) + vertex_offsets[src_type],
                torch.arange(num_dst_nodes) + vertex_offsets[dst_type],
            ])
            src_weight = torch.concat([src_weight, zeros(num_dst_nodes)])
            dst_weight = torch.concat([zeros(num_src_nodes), dst_weight])

        修复前 bug: 后两行在 else 块之外，src==dst 也会执行。
        """
        _dbg("NegSampleWeightBuilder._build_hetero_src_ne_dst",
             f"src_type='{src_type}' dst_type='{dst_type}' "
             f"num_src={num_src_nodes} num_dst={num_dst_nodes}")

        try:
            import torch
            vertices = torch.concat([
                torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
                + vertex_offsets[src_type],
                torch.arange(num_dst_nodes, dtype=torch.int64, device="cuda")
                + vertex_offsets[dst_type],
            ])
        except ImportError:
            # 无 torch 时用列表占位（单测用）
            src_off = vertex_offsets.get(src_type, 0)
            dst_off = vertex_offsets.get(dst_type, 0)
            vertices = (
                list(range(src_off, src_off + num_src_nodes)) +
                list(range(dst_off, dst_off + num_dst_nodes))
            )

        # 7ea1138 修复后: 权重 concat 仅在此分支执行
        src_weight_aligned = WeightAligner._pad_src_for_dst(
            src_weight, num_dst_nodes, weight_dtype
        )
        dst_weight_aligned = WeightAligner._pad_dst_for_src(
            dst_weight, num_src_nodes, weight_dtype
        )

        plan = NegSampleWeightPlan(
            vertices=vertices,
            src_weight=src_weight_aligned,
            dst_weight=dst_weight_aligned,
            src_dst_same_type=False,
            is_homogeneous=False,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
        )

        _dbg("NegSampleWeightBuilder._build_hetero_src_ne_dst",
             f"plan built: vertices_len={num_src_nodes + num_dst_nodes} "
             f"src_weight aligned to {num_src_nodes + num_dst_nodes} "
             f"dst_weight aligned to {num_src_nodes + num_dst_nodes}")
        return plan

    @staticmethod
    def _build_hetero_src_eq_dst(
        src_weight, dst_weight,
        num_src_nodes: int, num_dst_nodes: int,
        weight_dtype,
        vertex_offsets,
        node_type: str,       # input_type[0] == input_type[2]
    ) -> NegSampleWeightPlan:
        """
        对应 7ea1138 修复后的 `else:` 块（src==dst 类型）:

            vertices = (
                torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
                + graph_store._vertex_offsets[input_type[0]]
            )
            # src_weight 和 dst_weight 保持不变 ← 7ea1138 核心修复

        修复前 bug: src_weight 和 dst_weight 在此分支外被错误 concat，
                   导致 src_weight.shape = num_src + num_dst（应为 num_src）。
        """
        _dbg("NegSampleWeightBuilder._build_hetero_src_eq_dst",
             f"node_type='{node_type}' num_src={num_src_nodes} num_dst={num_dst_nodes} "
             f"[7ea1138 fix: weight NOT modified in this branch]")

        try:
            import torch
            vertices = (
                torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
                + vertex_offsets[node_type]
            )
        except ImportError:
            off = vertex_offsets.get(node_type, 0)
            vertices = list(range(off, off + num_src_nodes))

        # ← 7ea1138 修复: 权重不 concat，保持原始 num_src / num_dst 长度
        plan = NegSampleWeightPlan(
            vertices=vertices,
            src_weight=src_weight,   # 不变
            dst_weight=dst_weight,   # 不变
            src_dst_same_type=True,
            is_homogeneous=False,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
        )

        _dbg("NegSampleWeightBuilder._build_hetero_src_eq_dst",
             f"plan built: vertices_len={num_src_nodes} "
             f"src_weight UNCHANGED (len={num_src_nodes}) "
             f"dst_weight UNCHANGED (len={num_dst_nodes}) "
             f"← correct post-7ea1138")
        return plan

    @staticmethod
    def _build_homo(
        src_weight, dst_weight,
        num_src_nodes: int, num_dst_nodes: int,
        weight_dtype,
    ) -> NegSampleWeightPlan:
        """
        对应 7ea1138 修复后的 `else:` 最外层块（同构图）:

            vertices = torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")

        注意: 7ea1138 同时删除了:
            elif src_weight is None and dst_weight is None:
                vertices = None
        此分支为死代码（src_weight/dst_weight 已在前文被 ones() 填充）。
        WeightAligner._is_dead_branch() 文档化此事实。
        """
        _dbg("NegSampleWeightBuilder._build_homo",
             f"num_src={num_src_nodes} num_dst={num_dst_nodes} "
             f"[7ea1138: dead elif branch removed]")

        # 防御: 确认 7ea1138 删除的死代码确实不会到达
        WeightAligner._is_dead_branch(src_weight, dst_weight)

        try:
            import torch
            vertices = torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
        except ImportError:
            vertices = list(range(num_src_nodes))

        plan = NegSampleWeightPlan(
            vertices=vertices,
            src_weight=src_weight,
            dst_weight=dst_weight,
            src_dst_same_type=True,
            is_homogeneous=True,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
        )

        _dbg("NegSampleWeightBuilder._build_homo",
             f"plan built: vertices=arange({num_src_nodes})")
        return plan

    @classmethod
    def build(
        cls,
        src_weight,
        dst_weight,
        num_src_nodes: int,
        num_dst_nodes: int,
        weight_dtype,
        is_homogeneous: bool,
        src_type: Optional[str],   # input_type[0]; None if homogeneous
        dst_type: Optional[str],   # input_type[2]; None if homogeneous
        vertex_offsets: Optional[dict] = None,  # graph_store._vertex_offsets
    ) -> NegSampleWeightPlan:
        """
        顶层入口，对应 neg_sample() 中的 if/else 决策树（7ea1138 修复后）。

        调用路径:
          is_homogeneous=False, src_type!=dst_type → _build_hetero_src_ne_dst()
          is_homogeneous=False, src_type==dst_type → _build_hetero_src_eq_dst()
          is_homogeneous=True                      → _build_homo()

        断点调试: 打印决策入口 + 最终 plan shape
        """
        _dbg("NegSampleWeightBuilder.build",
             f"is_homo={is_homogeneous} src_type='{src_type}' dst_type='{dst_type}' "
             f"num_src={num_src_nodes} num_dst={num_dst_nodes} dtype={weight_dtype}")

        if not is_homogeneous:
            assert src_type is not None and dst_type is not None, (
                "src_type/dst_type 必须在异构图场景中提供"
            )
            assert vertex_offsets is not None, (
                "vertex_offsets 必须在异构图场景中提供"
            )
            if src_type != dst_type:
                plan = cls._build_hetero_src_ne_dst(
                    src_weight, dst_weight,
                    num_src_nodes, num_dst_nodes,
                    weight_dtype, vertex_offsets,
                    src_type, dst_type,
                )
            else:
                plan = cls._build_hetero_src_eq_dst(
                    src_weight, dst_weight,
                    num_src_nodes, num_dst_nodes,
                    weight_dtype, vertex_offsets,
                    src_type,
                )
        else:
            plan = cls._build_homo(
                src_weight, dst_weight,
                num_src_nodes, num_dst_nodes,
                weight_dtype,
            )

        # validate: 检查 shape 一致性，7ea1138 bug 会在此被捕获
        plan.validate()

        _dbg("NegSampleWeightBuilder.build",
             f"plan ready: is_homo={plan.is_homogeneous} "
             f"src_dst_same={plan.src_dst_same_type}")
        return plan


# ─── 便利函数: 直接对应 neg_sample() 中的调用点 ──────────────────────────────
def prepare_neg_sample_weights(
    src_weight,
    dst_weight,
    num_src_nodes: int,
    num_dst_nodes: int,
    weight_dtype,
    is_homogeneous: bool,
    input_type=None,          # Tuple[str, str, str] 或 None（同构时）
    vertex_offsets=None,      # graph_store._vertex_offsets
) -> NegSampleWeightPlan:
    """
    prepare_neg_sample_weights — neg_sample() 中 7ea1138 修复的顶层调用点。

    对应 neg_sample() 中 "# If the graph is heterogeneous, the weights need to
    be concatenated together and offsetted." 注释之后的全部逻辑。

    使用方式（对应 neg_sample() 调用）:
        plan = prepare_neg_sample_weights(
            src_weight=src_weight, dst_weight=dst_weight,
            num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes,
            weight_dtype=weight_dtype,
            is_homogeneous=graph_store.is_homogeneous,
            input_type=input_type,
            vertex_offsets=graph_store._vertex_offsets,
        )
        src_neg, dst_neg = _call_plc_negative_sampling(
            graph_store, num_neg, plan.vertices, plan.src_weight, plan.dst_weight
        )

    断点调试（WALPURGIS_DEBUG=1）打印全链路:
        1. build() 入口
        2. 分支选择（hetero_src_ne_dst / hetero_src_eq_dst / homo）
        3. weight concat 前后 shape（仅 src!=dst 分支）
        4. plan.validate() shape 一致性检查
    """
    _dbg("prepare_neg_sample_weights",
         f"ENTRY: is_homo={is_homogeneous} "
         f"input_type={input_type} "
         f"num_src={num_src_nodes} num_dst={num_dst_nodes}")

    src_type = input_type[0] if input_type is not None else None
    dst_type = input_type[2] if input_type is not None else None

    return NegSampleWeightBuilder.build(
        src_weight=src_weight,
        dst_weight=dst_weight,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        weight_dtype=weight_dtype,
        is_homogeneous=is_homogeneous,
        src_type=src_type,
        dst_type=dst_type,
        vertex_offsets=vertex_offsets,
    )
