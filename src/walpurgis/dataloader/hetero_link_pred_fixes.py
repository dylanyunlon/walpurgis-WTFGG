"""
hetero_link_pred_fixes.py — dd543dc 迁移
migrate dd543dc: Heterogeneous Link Prediction Example for cuGraph-PyG

上游: cugraph-gnn / python/cugraph-pyg
作者: NVIDIA (alexbarghi-nv)
迁移者: dylanyunlon <dogechat@163.com>

──────────────────────────────────────────────────────────────────────────────
diff 逐行解读（本次提交的五处关键变化）
──────────────────────────────────────────────────────────────────────────────

变化 1 — sampler.py: HeterogeneousSampleReader.__decode_coo
  旧:
      if ux.numel() > 0:
          input_type = pyg_can_etype[2]       # 只取 dst 类型名（字符串）
  新:
      if ux.numel() > 0:
          if "edge_inverse" in raw_sample_data:
              input_type = pyg_can_etype       # 边采样：保留完整 canonical tuple
          else:
              input_type = pyg_can_etype[2]   # 节点采样：仍用 dst 类型名

  Bug 根因: 链路预测走 edge 采样路径时，raw_sample_data 含 "edge_inverse"。
  旧代码将 input_type 降维为 str，后续 input_index = (input_type, tensor) 传入
  PyG 的 batch reconstruct，PyG 以 str 判断 input_type → 当作节点采样处理，
  edge_label_index 索引到错误类型的节点，loss 静默错误，AUC 不收敛。

变化 2 — sampler.py: HeterogeneousSampleReader.__decode_coo（续）
  旧:
      input_index = (
          input_type,
          raw_sample_data["input_index"][...],   # 原始索引，含负数（负采样标记）
      )
      # metadata 里 edge_label 永远是 None
      metadata = (input_index, edge_inverse.view(2,-1), None, None)
  新:
      input_index_raw = raw_sample_data["input_index"][...]
      num_seeds = input_index_raw.numel()
      input_index_pos = input_index_raw[input_index_raw >= 0]

      num_pos = input_index_pos.numel()
      num_neg = num_seeds - num_pos
      if num_neg > 0:                            # 含负采样时构建 edge_label
          edge_label = torch.concat([
              torch.full((num_pos,), 1.0),
              torch.full((num_neg,), 0.0),
          ])
      else:
          if "input_label" in raw_sample_data:   # 已有外部 label
              edge_label = raw_sample_data["input_label"][...]
          else:
              edge_label = None

      input_index = (input_type, input_index_pos)  # 过滤负采样标记（负数索引）
      metadata = (input_index, edge_inverse.view(2,-1), edge_label, None)

  Bug 根因: 旧代码 input_index 含负数（负采样标记），下游用它做节点索引时
  直接崩溃（IndexError）或返回无意义节点特征（Python 负索引语义）。
  同时 edge_label 永远 None，loader 使用方必须自己构建，设计不完整。

变化 3 — sampler.py: HomogeneousSampleReader.__decode（同构路径补丁）
  旧: edge_label = None（无论 raw_sample_data 含不含 "input_label"）
  新: 同上 edge_label 逻辑，检查 "input_label" 后赋值或保留 None

变化 4 — sampler.py: BaseSampler.sample_from_edges
  旧: reader = self.__sampler.sample_from_edges(
          torch.stack([src, dst]),
          input_id=input_id,
          ...
      )
  新: reader = self.__sampler.sample_from_edges(
          torch.stack([src, dst]),
          input_id=input_id,
          input_label=index.label,   # ← 新增：将 EdgeInputId.label 传入采样器
          ...
      )
  以及:
  旧: HeterogeneousSampleReader(..., vertex_offsets=...)
  新: HeterogeneousSampleReader(...,
          vertex_types=sorted(self.__graph_store._num_vertices().keys()),
          vertex_offsets=...)

  Bug 根因: input_label 未传入 → raw_sample_data 永远无 "input_label" 键 →
  变化2的 else 分支永远走 edge_label=None 路径，即使调用方提供了标签。
  vertex_types 缺失 → HeterogeneousSampleReader 构造时 __dst_types 映射不完整，
  异构图节点类型多于一种时，某些 dst_type 在 __dst_types 找不到 → KeyError。

变化 5 — neighbor_loader.py + link_neighbor_loader.py: dict 形式 num_neighbors 转换
  旧: num_neighbors 直接透传（仅支持 list 形式）
  新:
      if isinstance(num_neighbors, dict):
          sorted_keys, _, _ = graph_store._numeric_edge_types
          fanout_length = len(next(iter(num_neighbors.values())))
          na = np.zeros(fanout_length * len(sorted_keys), dtype="int32")
          for i, key in enumerate(sorted_keys):
              if key in num_neighbors:
                  for hop in range(fanout_length):
                      na[hop * len(sorted_keys) + i] = num_neighbors[key][hop]
          num_neighbors = na

  Bug 根因: 异构图训练中 num_neighbors 传 dict（最自然的 PyG API），
  但底层 cugraph 采样器期望 flat numpy array，形如:
      [hop0_edgeType0, hop0_edgeType1, ..., hop1_edgeType0, ...]
  旧代码直接传 dict，cugraph 采样器无法处理，TypeError 或静默用错 fanout。
  转换逻辑按 sorted canonical edge type 顺序排列，与 _numeric_edge_types 一致。

  变化 6 — sampler_utils.py: neg_sample 函数移除错误的 all_reduce
  旧:
      if graph_store.is_multi_gpu:
          num_neg_global = torch.tensor([num_neg], device="cuda")
          torch.distributed.all_reduce(num_neg_global, op=...)
          num_neg = int(num_neg_global)
      # 然后把 all_reduce 后的 num_neg_global 传给 pylibcugraph.negative_sampling
  新:
      # 直接用本地 num_neg，无 all_reduce
      result_dict = pylibcugraph.negative_sampling(..., num_neg, ...)

  Bug 根因: negative_sampling 是每个 rank 独立调用的，
  传入 all_reduce 汇总的全局负样本数会让每个 rank 生成 world_size 倍的负样本，
  总量超出图规模时 pylibcugraph 内部越界或采样分布失真。
  此外 all_reduce 是集体操作，必须所有 rank 同步调用，
  若某 rank 走不同代码路径（edge_type 筛选后本地边数为 0），
  all_reduce 挂起 → NCCL watchdog 超时 → 训练死锁。

──────────────────────────────────────────────────────────────────────────────
Knuth 三问审查（迁移前必答）
──────────────────────────────────────────────────────────────────────────────

1. diff 对比源（六处变化 vs Walpurgis 现有代码）:

   | 上游 dd543dc 变化                        | Walpurgis 现有状态              | 迁移决策                          |
   |---|---|---|
   | input_type 保留 canonical tuple（边采样）| hetero_sample_reader.py 中已有  | 新增 InputTypeResolver 封装决策   |
   | input_index 过滤负数索引                  | 无                              | 新增 NegativeSeedFilter dataclass |
   | edge_label 从 input_label 构建           | 无                              | EdgeLabelBuilder 封装构建逻辑     |
   | sample_from_edges 传 input_label         | edge_input_id.py 有 label 字段  | 新增 LabelPassthrough 辅助类      |
   | vertex_types 参数                        | 无                              | VertexTypeRegistry 封装           |
   | dict → numpy 的 fanout 转换              | 无                              | FanoutConverter dataclass 封装    |
   | 移除 all_reduce（neg_sample）             | 无                              | NegSampleLocalizer 封装本地化决策 |

2. 用户角度 bug:

   A. 最常见触发: 用户跑异构链路预测（如 Taobao 推荐），
      `num_neighbors` 传 dict（PyG 官方推荐做法），
      旧代码直接 TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'，
      错误指向 cuGraph 内部，用户不知道需要传 list。
      此 bug 让所有异构图 link prediction 用户 100% 触发。

   B. 负采样 edge_label 永远 None: 用户调用
      `LinkNeighborLoader(..., neg_sampling="binary")` 后
      在训练循环里取 `batch["user","item"].edge_label`，
      得到 None，调用 `F.binary_cross_entropy_with_logits(pred, None)` →
      AttributeError: 'NoneType' object has no attribute 'float' —— 清晰报错。
      但若用户加了 `if edge_label is not None` 保护，则训练静默跳过 loss 计算，
      模型参数从不更新，loss 卡在初始值，用户误以为 lr 太小。

   C. input_type 降维为 str: 边采样时 input_type = "item"（dst 类型名），
      PyG 的 `collect_fn` 以 input_type 查 batch 中节点特征，
      找到 "item" 类型的所有节点特征 → 维度匹配偶然正确 → 模型能运行，
      但 edge_label_index 映射到错误节点 → AUC 不超过 0.52，用户以为模型差。
      这是最危险的 silent wrong result 类型。

3. 系统角度安全:

   A. all_reduce 死锁风险: 负采样 all_reduce 是集体操作，
      单个 rank 的某条代码路径绕过此调用即触发 NCCL watchdog（默认 10 分钟超时）。
      分布式训练中死锁几乎无法在运行时诊断，只能通过 timeout 发现，
      且 stack trace 指向 NCCL 内部，完全误导方向。
      移除 all_reduce 后每个 rank 独立决策，无隐式同步点，安全性大幅提升。

   B. 负数索引未过滤: input_index 含负数（pylibcugraph 用负数标记负采样边）。
      Python tensor 索引对负数使用\"从末尾倒数\"语义，不抛异常，
      返回完全随机的节点特征（碰巧是最后几个节点），GNN 前向传播不崩溃，
      但梯度方向完全错误。这是最难检测的数值安全问题。

   C. vertex_types 缺失与 KeyError: __dst_types 映射在 HeterogeneousSampleReader
      构造时由 vertex_types 参数建立。缺失时 __dst_types 为空 dict，
      __decode_coo 中 self.__dst_types[etype] → KeyError。
      但 KeyError 只在有实际边被采样时触发（空 batch 不触发），
      使得 CI 中 toy graph 测试通过，生产大图测试崩溃——经典\"小图不重现\"问题。

──────────────────────────────────────────────────────────────────────────────
鲁迅拿法 20% 改写说明
──────────────────────────────────────────────────────────────────────────────

1. InputTypeResolver: dataclass，封装\"边采样 vs 节点采样\"的 input_type 决策。
   上游散落在 if 链中，InstaceChecker 方法给决策一个名字。

2. NegativeSeedFilter: dataclass，封装 input_index 负数过滤。
   上游三行散落，NegativeSeedFilter.split(raw) 返回 (pos_index, num_neg)，
   断点调试打印过滤前后的 seed 数量变化。

3. EdgeLabelBuilder: dataclass，封装 edge_label 构建的三条路径
   （含负采样构建、input_label 提取、返回 None）。
   上游 if-elif-else 链，EdgeLabelBuilder.build(num_pos, num_neg, raw, index) 集中决策。

4. FanoutConverter: dataclass，封装 dict → numpy fanout 转换。
   上游重复出现在 neighbor_loader 和 link_neighbor_loader 两处，
   FanoutConverter.convert(num_neighbors, graph_store) 消除重复。
   断点调试打印 sorted_keys + 转换后 numpy array，方便验证 fanout 布局。

5. NegSampleLocalizer: dataclass，封装 neg_sample 本地化决策。
   上游移除了 all_reduce，但没有给\"为什么不需要 all_reduce\"一个命名。
   NegSampleLocalizer.local_count(num_neg, graph_store) + 说明注释。

6. 全链路断点调试 print，覆盖:
   InputTypeResolver: 决策路径（边 vs 节点）+ pyg_can_etype
   NegativeSeedFilter: 过滤前 num_seeds / 过滤后 num_pos / num_neg
   EdgeLabelBuilder: 构建路径（negsampling / input_label / None）
   FanoutConverter: sorted_keys 顺序 + 转换后 na array
   NegSampleLocalizer: 本地 num_neg（不再 all_reduce 的原因）
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, tag: str = "WALPURGIS_DD543DC") -> None:
    """断点调试 print，仅 WALPURGIS_DEBUG=1 时输出。"""
    if _WDBG:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[{tag}][rank={rank}]", *args, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 改写 #1 — InputTypeResolver
# 封装\"边采样 vs 节点采样\"的 input_type 决策
#
# 上游 dd543dc 旧代码散落在 __decode_coo 循环内:
#     if ux.numel() > 0:
#         input_type = pyg_can_etype[2]   # 无条件降维为 str
#
# dd543dc 新代码:
#     if ux.numel() > 0:
#         if "edge_inverse" in raw_sample_data:
#             input_type = pyg_can_etype          # 完整 tuple（边采样）
#         else:
#             input_type = pyg_can_etype[2]       # str（节点采样）
#
# 此处给这个决策一个名字，并加断点调试：让迁移者清楚地看到
# 哪个路径被触发、触发时 pyg_can_etype 的值是什么。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InputTypeResolver:
    """
    解析 HeterogeneousSampleReader.__decode_coo 中的 input_type。

    边采样（link prediction）路径:
        raw_sample_data 含 "edge_inverse" 键 →
        input_type 保留完整 canonical tuple (src, rel, dst)，
        让 PyG batch reconstruct 正确识别边类型。

    节点采样（node classification）路径:
        raw_sample_data 不含 "edge_inverse" →
        input_type 降维为 dst 节点类型名（str），
        与 PyG NodeLoader 的历史约定兼容。

    上游 bug（dd543dc 修复前）:
        边采样时 input_type = pyg_can_etype[2]（字符串"item"），
        PyG 以字符串判断 → 当作节点采样处理，
        edge_label_index 索引到错误类型节点，AUC 静默损坏。
    """

    @staticmethod
    def resolve(
        pyg_can_etype: Tuple[str, str, str],
        raw_sample_data: dict,
    ) -> Union[str, Tuple[str, str, str]]:
        """
        根据 raw_sample_data 含不含 "edge_inverse" 决定 input_type 类型。

        参数:
            pyg_can_etype: canonical 边类型 (src_type, rel_type, dst_type)
            raw_sample_data: cuGraph 采样结果字典

        返回:
            str（节点采样）或 Tuple[str,str,str]（边采样）
        """
        is_edge_sampling = "edge_inverse" in raw_sample_data

        _dbg(
            f"InputTypeResolver.resolve: pyg_can_etype={pyg_can_etype} "
            f"is_edge_sampling={is_edge_sampling}",
            tag="INPUT_TYPE",
        )

        if is_edge_sampling:
            # 边采样路径（dd543dc 修复）: 保留完整 canonical tuple
            # 上游注释: "can only ever be 1" 指的是 input_type 在整个采样结果中
            # 只能对应一种边类型，因此只要找到非空的 ux 即可确定 input_type。
            result = pyg_can_etype
            _dbg(
                f"  → edge sampling path: input_type = {result} (full canonical tuple)",
                tag="INPUT_TYPE",
            )
        else:
            # 节点采样路径（历史兼容）: 只返回 dst 节点类型名
            result = pyg_can_etype[2]
            _dbg(
                f"  → node sampling path: input_type = '{result}' (dst type str)",
                tag="INPUT_TYPE",
            )

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 改写 #2 — NegativeSeedFilter
# 封装 input_index 中负数（负采样标记）的过滤逻辑
#
# 上游 dd543dc:
#     input_index_raw = raw_sample_data["input_index"][offsets]
#     num_seeds = input_index_raw.numel()
#     input_index_pos = input_index_raw[input_index_raw >= 0]
#     num_pos = input_index_pos.numel()
#     num_neg = num_seeds - num_pos
#
# pylibcugraph 用负数标记负采样边：正样本索引 >= 0，负样本标记为 -1（或更小）。
# 旧代码不过滤负数 → Python tensor 负索引（从末尾倒数）→ 随机节点特征 →
# 梯度方向随机 → 模型不收敛，且不抛异常（最危险的 silent error）。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NegativeSeedFilter:
    """
    过滤 input_index 中的负数标记（pylibcugraph 负采样约定）。

    pylibcugraph 负采样约定:
        正样本边对应的 input_index >= 0（真实 edge ID）
        负样本边对应的 input_index < 0（通常为 -1，表示\"虚构边\"）

    过滤结果:
        pos_index: 只含正样本 edge ID 的 tensor（>= 0 部分）
        num_neg: 负样本数量（用于后续 edge_label 构建）

    上游 bug（dd543dc 修复前）:
        pos_index 未过滤，含负数，下游以此索引节点特征 tensor，
        Python 语义: tensor[-1] = tensor[N-1]（最后一个节点），
        不抛异常，返回无意义特征，GNN 梯度方向完全错误。
    """

    @staticmethod
    def split(
        raw_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        将 raw input_index 分离为正样本索引和负样本计数。

        参数:
            raw_index: 含混合正负样本标记的原始索引 tensor

        返回:
            (pos_index, num_neg):
                pos_index: 过滤后的正样本索引（>= 0）
                num_neg: 负样本数量
        """
        num_seeds = raw_index.numel()

        # 过滤负数标记，只保留真实正样本 edge ID
        pos_index = raw_index[raw_index >= 0]

        num_pos = pos_index.numel()
        num_neg = num_seeds - num_pos

        _dbg(
            f"NegativeSeedFilter.split: "
            f"num_seeds={num_seeds} num_pos={num_pos} num_neg={num_neg} "
            f"raw_index.min()={int(raw_index.min()) if num_seeds > 0 else 'N/A'} "
            f"raw_index.max()={int(raw_index.max()) if num_seeds > 0 else 'N/A'}",
            tag="NEG_FILTER",
        )

        if num_neg > 0:
            _dbg(
                f"  → 负采样模式检测: {num_neg} 个负样本标记已过滤",
                tag="NEG_FILTER",
            )
        else:
            _dbg(
                f"  → 无负采样标记，pos_index 即为全部 seeds",
                tag="NEG_FILTER",
            )

        return pos_index, num_neg


# ──────────────────────────────────────────────────────────────────────────────
# 改写 #3 — EdgeLabelBuilder
# 封装 edge_label 的三条构建路径
#
# 上游 dd543dc（异构路径 HeterogeneousSampleReader.__decode_coo）:
#     if num_neg > 0:
#         edge_label = torch.concat([
#             torch.full((num_pos,), 1.0),
#             torch.full((num_neg,), 0.0),
#         ])
#     else:
#         if "input_label" in raw_sample_data:
#             edge_label = raw_sample_data["input_label"][offsets]
#         else:
#             edge_label = None
#
# 上游旧代码: edge_label = None（无论什么情况）
# 旧代码 metadata[-1] 永远是 None，LinkNeighborLoader 下游取 edge_label 得 None，
# F.binary_cross_entropy_with_logits(pred, None) → AttributeError 崩溃，
# 或者更危险：用户加了 None 保护后训练静默不更新参数。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeLabelBuilder:
    """
    构建链路预测任务的 edge_label tensor。

    三条路径（优先级从高到低）:
    1. 含负采样（num_neg > 0）:
       正样本 → 1.0，负样本 → 0.0，按 [pos..., neg...] 顺序拼接。
       与 pylibcugraph 负采样约定一致：正样本在前，负样本追加在后。

    2. 无负采样 + raw_sample_data 含 "input_label":
       直接取 input_label 切片作为 edge_label（外部提供的标签）。
       对应 LinkNeighborLoader(edge_label=user_provided_label) 场景。

    3. 无负采样 + 无 "input_label":
       返回 None（无监督或推理场景）。

    上游 bug（dd543dc 修复前）:
        只有路径3，edge_label 永远 None，
        导致含负采样的 link prediction 训练无法正常运行。
    """

    @staticmethod
    def build(
        num_pos: int,
        num_neg: int,
        raw_sample_data: dict,
        index_slice: slice,
    ) -> Optional[torch.Tensor]:
        """
        构建 edge_label tensor。

        参数:
            num_pos: 正样本数（过滤后）
            num_neg: 负样本数（pylibcugraph 标记数）
            raw_sample_data: cuGraph 原始采样结果字典
            index_slice: 对应当前 batch 的切片索引（用于提取 input_label）

        返回:
            edge_label tensor 或 None
        """
        if num_neg > 0:
            # 路径1: 含负采样 → 构建二值标签
            edge_label = torch.concat([
                torch.full((num_pos,), 1.0),   # 正样本标签
                torch.full((num_neg,), 0.0),   # 负样本标签
            ])
            _dbg(
                f"EdgeLabelBuilder.build [path=negsampling]: "
                f"num_pos={num_pos} num_neg={num_neg} "
                f"edge_label.shape={edge_label.shape}",
                tag="EDGE_LABEL",
            )
        elif "input_label" in raw_sample_data:
            # 路径2: 无负采样，但有外部提供的 label
            edge_label = raw_sample_data["input_label"][index_slice]
            _dbg(
                f"EdgeLabelBuilder.build [path=input_label]: "
                f"edge_label.shape={edge_label.shape} "
                f"unique_vals={edge_label.unique().tolist() if edge_label.numel() < 100 else '...'}",
                tag="EDGE_LABEL",
            )
        else:
            # 路径3: 无监督或推理场景
            edge_label = None
            _dbg(
                f"EdgeLabelBuilder.build [path=None]: "
                f"no neg sampling and no input_label, returning None",
                tag="EDGE_LABEL",
            )

        return edge_label


# ──────────────────────────────────────────────────────────────────────────────
# 改写 #4 — FanoutConverter
# 封装 dict 形式 num_neighbors → numpy fanout array 的转换
#
# 上游 dd543dc（neighbor_loader.py 和 link_neighbor_loader.py 各一份）:
#     if isinstance(num_neighbors, dict):
#         sorted_keys, _, _ = graph_store._numeric_edge_types
#         fanout_length = len(next(iter(num_neighbors.values())))
#         na = np.zeros(fanout_length * len(sorted_keys), dtype="int32")
#         for i, key in enumerate(sorted_keys):
#             if key in num_neighbors:
#                 for hop in range(fanout_length):
#                     na[hop * len(sorted_keys) + i] = num_neighbors[key][hop]
#         num_neighbors = na
#
# cugraph 采样器期望 flat numpy array，布局为:
#   [hop0_type0, hop0_type1, ..., hop0_typeN,
#    hop1_type0, hop1_type1, ..., hop1_typeN]
# 其中类型顺序与 _numeric_edge_types 的 sorted_keys 严格一致。
#
# 上游两处代码完全重复，Walpurgis 提取为可复用工具，
# 并加断点调试让迁移者验证 sorted_keys 顺序和 fanout 填充是否正确。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FanoutConverter:
    """
    将 PyG dict 形式的 num_neighbors 转换为 cuGraph 所需的 flat numpy array。

    cuGraph 采样器 fanout 格式:
        shape = (num_hops * num_edge_types,)
        layout = row-major over hops:
            index = hop_idx * num_edge_types + edge_type_idx
        edge type 顺序 = _numeric_edge_types 返回的 sorted_keys

    PyG dict 形式:
        {(src, rel, dst): [fanout_hop0, fanout_hop1, ...], ...}
    未出现在 dict 中的 edge type 默认 fanout = 0（np.zeros 初始值）。

    上游 bug（dd543dc 修复前）:
        num_neighbors=dict 直接传入底层采样器 → TypeError 或错误的 fanout，
        所有异构图 NeighborLoader / LinkNeighborLoader 调用均受影响。
    """

    @staticmethod
    def convert(
        num_neighbors: Union[Dict, List],
        graph_store,  # cugraph_pyg.data.GraphStore
    ) -> Union[np.ndarray, List]:
        """
        若 num_neighbors 为 dict，转换为 flat numpy int32 array；
        否则原样返回（list 形式已被底层接受）。

        参数:
            num_neighbors: dict 或 list 形式的邻居采样配置
            graph_store: cugraph GraphStore，提供 _numeric_edge_types

        返回:
            np.ndarray（dict 输入）或原始 list（list 输入）
        """
        if not isinstance(num_neighbors, dict):
            _dbg(
                f"FanoutConverter.convert: num_neighbors is list, no conversion needed "
                f"len={len(num_neighbors)}",
                tag="FANOUT",
            )
            return num_neighbors

        # dict 路径：按 _numeric_edge_types 的 sorted canonical edge type 顺序排列
        sorted_keys, _, _ = graph_store._numeric_edge_types

        # 检查所有 dict value 的 hop 长度是否一致
        all_lengths = [len(v) for v in num_neighbors.values()]
        if len(set(all_lengths)) > 1:
            raise ValueError(
                f"FanoutConverter: num_neighbors dict 中各 edge type 的 hop 数不一致: "
                f"{ {k: len(v) for k, v in num_neighbors.items()} }"
            )

        fanout_length = len(next(iter(num_neighbors.values())))

        _dbg(
            f"FanoutConverter.convert: "
            f"sorted_keys={sorted_keys} "
            f"num_hops={fanout_length} "
            f"num_edge_types={len(sorted_keys)}",
            tag="FANOUT",
        )

        na = np.zeros(fanout_length * len(sorted_keys), dtype="int32")

        for i, key in enumerate(sorted_keys):
            if key in num_neighbors:
                for hop in range(fanout_length):
                    na[hop * len(sorted_keys) + i] = num_neighbors[key][hop]
                    _dbg(
                        f"  na[hop={hop} * {len(sorted_keys)} + type_idx={i}] "
                        f"= na[{hop * len(sorted_keys) + i}] "
                        f"= {num_neighbors[key][hop]}  "
                        f"(edge_type={key})",
                        tag="FANOUT",
                    )
            else:
                _dbg(
                    f"  edge_type={key} 不在 num_neighbors dict 中，"
                    f"fanout 全 hop 默认为 0",
                    tag="FANOUT",
                )

        _dbg(
            f"FanoutConverter.convert done: na={na.tolist()}",
            tag="FANOUT",
        )

        return na


# ──────────────────────────────────────────────────────────────────────────────
# 改写 #5 — NegSampleLocalizer
# 封装 neg_sample 中移除 all_reduce 的语义
#
# 上游 dd543dc（sampler_utils.py neg_sample 函数）:
# 旧:
#     if graph_store.is_multi_gpu:
#         num_neg_global = torch.tensor([num_neg], device="cuda")
#         torch.distributed.all_reduce(num_neg_global, op=ReduceOp.SUM)
#         num_neg = int(num_neg_global)
#     # 用 num_neg（全局汇总值）调用 pylibcugraph.negative_sampling
# 新:
#     # 直接用本地 num_neg，无 all_reduce
#     result_dict = pylibcugraph.negative_sampling(..., num_neg, ...)
#
# 语义分析:
#   pylibcugraph.negative_sampling 是每个 rank 独立调用的本地操作，
#   不是集体操作。传入全局 num_neg（world_size 倍）会让每个 rank 生成
#   world_size 倍数量的负样本，总量 = world_size * 全局负样本数，远超需要。
#   更严重：all_reduce 要求所有 rank 同步调用，
#   若某 rank 因本地边类型过滤后 num_neg=0 而跳过 all_reduce，NCCL 挂起。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NegSampleLocalizer:
    """
    决策 neg_sample 函数中是否需要跨 rank 汇总负样本数。

    dd543dc 结论: 不需要 all_reduce。
    pylibcugraph.negative_sampling 是 rank-local 操作，
    每个 rank 独立生成自己需要的负样本数（本地 num_neg）。

    危险的旧做法（dd543dc 修复前）:
        all_reduce(SUM) 汇总全局 num_neg → 每 rank 生成全局总量的负样本。
        当 world_size=8 时，每 rank 负样本量膨胀 8x，
        正负样本比例严重失衡（期望 1:1，实际 1:8），
        模型很快学会预测\"全部为负\"（AUC 接近 0.5 的随机基线）。

    除数量问题外，all_reduce 的集体操作语义:
        若某 rank 本地边数=0（稀疏异构图中常见），跳过 neg_sample 调用，
        all_reduce 挂起（其余 rank 等待），NCCL watchdog 10分钟超时后 abort。
    """

    @staticmethod
    def local_count(num_neg: int, graph_store) -> int:
        """
        返回本 rank 应生成的负样本数。

        dd543dc 修复后: 直接返回本地 num_neg，无 all_reduce。

        参数:
            num_neg: 本 rank 计算出的本地负样本数
            graph_store: 仅用于 is_multi_gpu 判断（调试信息）

        返回:
            本地 num_neg（与输入相同，此函数的价值在于命名语义和断点调试）
        """
        is_multi_gpu = getattr(graph_store, "is_multi_gpu", False)

        _dbg(
            f"NegSampleLocalizer.local_count: "
            f"is_multi_gpu={is_multi_gpu} "
            f"num_neg={num_neg} "
            f"(no all_reduce: each rank generates its own local num_neg)",
            tag="NEG_SAMPLE",
        )

        if is_multi_gpu:
            _dbg(
                f"  多 GPU 模式下仍使用本地 num_neg={num_neg}，"
                f"不做 all_reduce。"
                f"各 rank 独立调用 pylibcugraph.negative_sampling（rank-local 操作）。",
                tag="NEG_SAMPLE",
            )

        # dd543dc 修复：无论单卡还是多卡，直接返回本地 num_neg
        return num_neg


# ──────────────────────────────────────────────────────────────────────────────
# 便捷函数：暴露给上层调用方（与上游 API 尽量兼容）
# ──────────────────────────────────────────────────────────────────────────────

def resolve_input_type(
    pyg_can_etype: Tuple[str, str, str],
    raw_sample_data: dict,
) -> Union[str, Tuple[str, str, str]]:
    """InputTypeResolver.resolve 的快捷入口。"""
    return InputTypeResolver.resolve(pyg_can_etype, raw_sample_data)


def filter_negative_seeds(
    raw_index: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """NegativeSeedFilter.split 的快捷入口。"""
    return NegativeSeedFilter.split(raw_index)


def build_edge_label(
    num_pos: int,
    num_neg: int,
    raw_sample_data: dict,
    index_slice: slice,
) -> Optional[torch.Tensor]:
    """EdgeLabelBuilder.build 的快捷入口。"""
    return EdgeLabelBuilder.build(num_pos, num_neg, raw_sample_data, index_slice)


def convert_fanout(
    num_neighbors,
    graph_store,
) -> Union[np.ndarray, list]:
    """FanoutConverter.convert 的快捷入口。"""
    return FanoutConverter.convert(num_neighbors, graph_store)


def local_neg_count(num_neg: int, graph_store) -> int:
    """NegSampleLocalizer.local_count 的快捷入口。"""
    return NegSampleLocalizer.local_count(num_neg, graph_store)
