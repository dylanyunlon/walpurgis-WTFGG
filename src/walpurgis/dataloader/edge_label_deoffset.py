"""
边标签索引去偏移修正模块
migrate 7c2907f: [BUG] Correct De-Offset of Edge Label Index
migrate 5baac8b: [BUG] Fix bug with drop_last when mod is 0

上游: cugraph-gnn / python/cugraph-pyg/cugraph_pyg/{loader,sampler}
作者: Alex Barghi (alexbarghi-nv)

Bug 根因 (7c2907f):
  HeterogeneousSampleReader.__decode_coo 中 edge_inverse 携带全局节点 offset。
  旧代码以 integer_input_type 索引 __vertex_offsets 做固定减法——
  当 src_type != dst_type 时两侧节点分属不同 offset 段，
  用同一套 __vertex_offsets 做减法，方向与量级均错误。
  修复: 不再依赖全局 offset 表，改用词典序比较边两端节点类型名，
  按 minibatch 内实际位置 (max+1) 动态去偏移。

Bug 根因 (5baac8b):
  NodeLoader / LinkLoader 的 __iter__ 中，drop_last=True 时执行:
      d = perm.numel() % self.__batch_size
      perm = perm[:-d]
  当 perm.numel() 恰好整除 batch_size 时，d == 0，
  Python 负索引 perm[:-0] 等价于 perm[:0]，返回空 tensor——
  训练集恰好被整除时，一批数据都不迭代，静默丢失全部样本。
  修复: 加 if d > 0 保护，d==0 时跳过切片，保留完整 perm。

鲁迅拿法改写20%:
  - EdgeInverseBundle: 值对象，携带 (src, dst, input_type) 三元组，
    替代上游裸 list [edge_inverse[0], edge_inverse[1]]
  - DeOffsetStrategy: 枚举，LEXICOGRAPHIC (7c2907f 新路) 与 VERTEX_OFFSET (旧 BUG 路)，
    build_deoffset_session 见到 VERTEX_OFFSET 直接 raise
  - HeteroEdgeLabelDeoffset: 执行类，封装词典序去偏移逻辑，可测试
  - InputTensorGuard: 封装 detach().clone() + drop_last 校验，
    对应 link_loader.py / node_loader.py 两处修复
  - HomoEdgeInverseView: 封装 view(2, -1) 提前赋值，对应 HomogeneousSampleReader 修复
  - PermDropLastSlicer: 封装 5baac8b 修复——drop_last 时对 perm 的安全裁剪，
    d==0 时不做切片，替代上游 NodeLoader/LinkLoader.__iter__ 裸条件

调试断点: 设 WALPURGIS_DEBUG=1 激活全链路 print
  1. InputTensorGuard.__init__ 入口
  2. InputTensorGuard._check_drop_last 校验
  3. HeteroEdgeLabelDeoffset.apply 入口 + 词典序判断
  4. HomoEdgeInverseView.apply 入口
  5. build_deoffset_session 工厂函数出口
  6. PermDropLastSlicer.apply 入口 + d 值 + 裁剪决策
"""

import os
import sys
import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ─────────────────────────────────────────────
# 1. 枚举: 去偏移策略
# ─────────────────────────────────────────────

class DeOffsetStrategy(enum.Enum):
    """
    两种去偏移路径的枚举。

    VERTEX_OFFSET: 上游旧代码（7c2907f 之前）——用 __vertex_offsets[integer_input_type]
                   固定减法。仅在 src_type == dst_type 时碰巧正确；
                   异构双向图中必然出错，已被废弃。
    LEXICOGRAPHIC: 7c2907f 修复路——比较 input_type[0] 与 input_type[2] 字符串词典序，
                   按 minibatch 内实际节点范围 (max+1) 动态去偏移。
    """
    VERTEX_OFFSET = "vertex_offset"   # BUG 路径，禁止使用
    LEXICOGRAPHIC = "lexicographic"   # 7c2907f 正确路径


# ─────────────────────────────────────────────
# 2. 值对象: 边逆向索引束
# ─────────────────────────────────────────────

@dataclass
class EdgeInverseBundle:
    """
    携带 heterogeneous link prediction 中 edge_inverse 的三元组。

    src: edge_inverse[0] — minibatch 内源节点全局编号（含 offset）
    dst: edge_inverse[1] — minibatch 内目标节点全局编号（含 offset）
    input_type: (src_type, rel_type, dst_type) 规范边类型元组
    """
    src: torch.Tensor
    dst: torch.Tensor
    input_type: Tuple[str, str, str]

    def __post_init__(self):
        if not isinstance(self.input_type, tuple) or len(self.input_type) != 3:
            raise ValueError(
                f"[EdgeInverseBundle] input_type 须为 3 元组 (src, rel, dst)，"
                f"实际: {self.input_type!r}"
            )
        if self.src.shape != self.dst.shape:
            raise ValueError(
                f"[EdgeInverseBundle] src.shape={self.src.shape} != "
                f"dst.shape={self.dst.shape}"
            )
        if _DBG:
            print(
                f"[WALPURGIS:EdgeInverseBundle] input_type={self.input_type} "
                f"src.shape={self.src.shape} "
                f"src.max={self.src.max().item() if self.src.numel() > 0 else 'N/A'} "
                f"dst.max={self.dst.max().item() if self.dst.numel() > 0 else 'N/A'}",
                file=sys.stderr,
            )


# ─────────────────────────────────────────────
# 3. 执行类: heterogeneous 去偏移
# ─────────────────────────────────────────────

class HeteroEdgeLabelDeoffset:
    """
    7c2907f 核心修复: 词典序去偏移。

    上游原始逻辑（有 BUG）::

        edge_inverse[0] -= self.__vertex_offsets[self.__src_types[integer_input_type]]
        edge_inverse[1] -= self.__vertex_offsets[self.__dst_types[integer_input_type]]

    这里 __vertex_offsets 是全图各类型节点的全局起始偏移——
    在 heterogeneous 图 renumber 后，同一 minibatch 内不同类型节点
    被 concat 为一段连续编号，offset 排列取决于词典序而非 integer_input_type。
    当 src_type < dst_type（词典序）时，src 在前，dst 段起始 = src 段大小 = src.max()+1；
    反之 dst 在前，src 段起始 = dst.max()+1。

    修复后逻辑（7c2907f）::

        if input_type[0] != input_type[2]:          # src_type != dst_type
            if input_type[0] < input_type[2]:       # 词典序: src 在前
                edge_inverse[1] -= edge_inverse[0].max() + 1
            else:                                   # 词典序: dst 在前
                edge_inverse[0] -= edge_inverse[1].max() + 1
        # 若 src_type == dst_type: 两端共享同一 offset 段，无需减法

    Walpurgis 改写: 封装为可测试类，in-place 修改 bundle.src / bundle.dst。
    """

    def __init__(self, strategy: DeOffsetStrategy = DeOffsetStrategy.LEXICOGRAPHIC):
        if strategy == DeOffsetStrategy.VERTEX_OFFSET:
            raise ValueError(
                "[HeteroEdgeLabelDeoffset] VERTEX_OFFSET 策略是 7c2907f 修复前的 BUG 路径，"
                "已禁止使用。请使用 DeOffsetStrategy.LEXICOGRAPHIC。"
            )
        self.strategy = strategy

    def apply(self, bundle: EdgeInverseBundle) -> EdgeInverseBundle:
        """
        对 bundle 执行词典序去偏移，in-place 修改 src/dst，返回同一 bundle。

        若 src_type == dst_type，skip（共享 offset 段，无需减法）。
        """
        src_type, _rel, dst_type = bundle.input_type

        if _DBG:
            print(
                f"[WALPURGIS:HeteroEdgeLabelDeoffset.apply] "
                f"src_type={src_type!r} dst_type={dst_type!r} strategy={self.strategy}",
                file=sys.stderr,
            )

        if src_type == dst_type:
            # 同类型节点，两端共享同一 offset 段，不做减法
            if _DBG:
                print(
                    f"[WALPURGIS:HeteroEdgeLabelDeoffset.apply] "
                    f"src_type==dst_type，跳过去偏移",
                    file=sys.stderr,
                )
            return bundle

        # 词典序判断: src_type < dst_type => src 排在 minibatch renumber map 前段
        if src_type < dst_type:
            # dst 段起始偏移 = src 段节点数 = src 编号最大值 + 1
            offset = bundle.src.max() + 1
            if _DBG:
                print(
                    f"[WALPURGIS:HeteroEdgeLabelDeoffset.apply] "
                    f"词典序 src<dst，dst -= src.max()+1={offset.item()}",
                    file=sys.stderr,
                )
            bundle.dst -= offset
        else:
            # src_type > dst_type: dst 排在前段，src 段起始偏移 = dst.max()+1
            offset = bundle.dst.max() + 1
            if _DBG:
                print(
                    f"[WALPURGIS:HeteroEdgeLabelDeoffset.apply] "
                    f"词典序 src>dst，src -= dst.max()+1={offset.item()}",
                    file=sys.stderr,
                )
            bundle.src -= offset

        return bundle


# ─────────────────────────────────────────────
# 4. 值对象: HomogeneousSampleReader view 提前
# ─────────────────────────────────────────────

class HomoEdgeInverseView:
    """
    7c2907f 对 HomogeneousSampleReader 的修复:
    将 edge_inverse.view(2, -1) 提前赋值给变量，再放入 metadata tuple，
    避免 metadata 中持有对未 view 的原始 tensor 的引用，
    确保后续对 metadata 的消费者拿到已经 reshape 的 tensor。

    上游旧代码::

        metadata = (input_index, edge_inverse.view(2, -1), ...)

    修复后::

        edge_inverse = edge_inverse.view(2, -1)
        metadata = (input_index, edge_inverse, ...)

    Walpurgis 封装: 纯函数，输入 1D edge_inverse tensor，输出 (2, N) tensor。
    """

    @staticmethod
    def apply(edge_inverse: torch.Tensor) -> torch.Tensor:
        """
        将 flat edge_inverse reshape 为 (2, N)，与 7c2907f 修复语义一致。
        同时做形状校验，防止 numel 为奇数时 view 崩溃留下难以定位的 RuntimeError。
        """
        if _DBG:
            print(
                f"[WALPURGIS:HomoEdgeInverseView.apply] "
                f"input shape={edge_inverse.shape} numel={edge_inverse.numel()}",
                file=sys.stderr,
            )

        if edge_inverse.numel() % 2 != 0:
            raise ValueError(
                f"[HomoEdgeInverseView] edge_inverse.numel()={edge_inverse.numel()} "
                f"不能整除 2，无法 view(2, -1)。"
                f"上游 input_offsets 切片可能有误。"
            )

        result = edge_inverse.view(2, -1)

        if _DBG:
            print(
                f"[WALPURGIS:HomoEdgeInverseView.apply] "
                f"output shape={result.shape}",
                file=sys.stderr,
            )

        return result


# ─────────────────────────────────────────────
# 5. 输入张量守卫: detach + clone + drop_last 校验
# ─────────────────────────────────────────────

class InputTensorGuard:
    """
    7c2907f 对 LinkLoader / NodeLoader 的两处修复封装:

    1. detach().clone():
       调用方传入的 edge_label_index / input_nodes 可能仍挂在计算图上，
       后续 __vertex_offsets 的 in-place 加法会静默修改原始 tensor，
       污染调用方的梯度图或缓存数据。
       detach().clone() 彻底断开，确保 Loader 持有独立副本。

    2. drop_last 早期校验:
       若 tensor 中元素数 < batch_size 且 drop_last=True，
       所有 batch 都会被丢弃，产生静默空结果而非明确报错。
       提前 raise ValueError，给用户清晰提示。

    Walpurgis 改写: 提取为可独立测试的守卫类，支持 edge_label_index 与 input_nodes 两种语义。
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        drop_last: bool,
        mode: str = "edge",  # "edge" | "node"
    ):
        if _DBG:
            print(
                f"[WALPURGIS:InputTensorGuard.__init__] "
                f"mode={mode} "
                f"raw_shape={tensor.shape} "
                f"batch_size={batch_size} drop_last={drop_last}",
                file=sys.stderr,
            )

        # 断开计算图，保护调用方 tensor 不被 in-place 修改
        self.tensor = tensor.detach().clone()
        self.mode = mode

        self._check_drop_last(batch_size, drop_last)

    def _check_drop_last(self, batch_size: int, drop_last: bool) -> None:
        """
        7c2907f 新增的早期校验:
        当 tensor 元素数不足一个 batch 且 drop_last=True 时，
        立即 raise，防止静默空结果。
        """
        if self.mode == "edge":
            # edge_label_index shape: (2, N), 边数 = shape[1]
            count = self.tensor.shape[1] if self.tensor.dim() == 2 else self.tensor.numel()
        else:
            count = self.tensor.numel()

        if _DBG:
            print(
                f"[WALPURGIS:InputTensorGuard._check_drop_last] "
                f"mode={self.mode} count={count} batch_size={batch_size} drop_last={drop_last}",
                file=sys.stderr,
            )

        if count < batch_size and drop_last:
            entity = "edges" if self.mode == "edge" else "nodes"
            param = "edge_label_index" if self.mode == "edge" else "input_nodes"
            raise ValueError(
                f"The number of input {entity} ({count}) is less than the batch size "
                f"({batch_size}) and drop_last is True. "
                f"This will result in all batches being dropped. "
                f"Either set drop_last to False or increase "
                f"the number of {entity} in {param}."
            )


# ─────────────────────────────────────────────
# 6. 工厂函数: 构建去偏移 session
# ─────────────────────────────────────────────

def build_deoffset_session(
    strategy: DeOffsetStrategy = DeOffsetStrategy.LEXICOGRAPHIC,
) -> HeteroEdgeLabelDeoffset:
    """
    工厂函数，对应 7c2907f 修复后的标准使用路径。

    Args:
        strategy: 去偏移策略枚举。传入 VERTEX_OFFSET 会直接 raise（旧 BUG 路径已废弃）。

    Returns:
        HeteroEdgeLabelDeoffset 实例。
    """
    if _DBG:
        print(
            f"[WALPURGIS:build_deoffset_session] strategy={strategy}",
            file=sys.stderr,
        )

    session = HeteroEdgeLabelDeoffset(strategy=strategy)

    if _DBG:
        print(
            f"[WALPURGIS:build_deoffset_session] session 已就绪: {session.__class__.__name__}",
            file=sys.stderr,
        )

    return session


# ─────────────────────────────────────────────
# 7. perm 裁剪守卫: drop_last mod==0 安全切片
# ─────────────────────────────────────────────

class PermDropLastSlicer:
    """
    5baac8b 核心修复: drop_last 模式下对排列张量 perm 的安全裁剪。

    上游旧代码 (NodeLoader / LinkLoader 各一处，共两处，逻辑相同):

        if self.__drop_last:
            d = perm.numel() % self.__batch_size
            perm = perm[:-d]          # BUG: d==0 时等价于 perm[:0] => 空 tensor

    Python 负索引语义:
        perm[:-d]  当 d > 0 时，从末尾丢弃 d 个元素 —— 正确
        perm[:-0]  Python 中 -0 == 0，等价于 perm[:0]  —— 返回空 tensor

    后果:
        当样本总数恰好整除 batch_size（例如 1024 样本 / batch_size=32），
        d == 0，所有 batch 迭代消失，训练/推理静默跑零步，
        loss 不更新，用户毫无提示，极难排查。

    修复后 (5baac8b):

        if self.__drop_last:
            d = perm.numel() % self.__batch_size
            if d > 0:                 # 整除时 d==0，跳过切片，保留完整 perm
                perm = perm[:-d]

    Walpurgis 改写:
        封装为独立可测试类。apply() 是纯函数语义（无 in-place），
        返回新 tensor（整除时原地引用，d>0 时切片副本）。
        NodeLoader 与 LinkLoader 共享同一实现，消除重复。

    调试断点 (WALPURGIS_DEBUG=1):
        - 入口: perm.numel(), batch_size, d 值
        - 决策: skip (d==0, 整除) 或 slice (d>0, 丢弃尾部 d 个)
        - 出口: 裁剪后 perm.numel()

    Knuth 审查备注:
        1. diff 对比源:
           上游两文件 (node_loader.py L138-140, link_loader.py L188-190) 各有一处
           相同的单行 `perm = perm[:-d]`，修复统一为 `if d > 0` 保护。
           Walpurgis 在此统一封装，避免未来出现第三处遗漏。

        2. 用户角度 bug:
           整除场景（如数据集大小是 batch_size 的整数倍）静默产生空迭代器，
           train_loop 跑零步，但不报错、不打 warning，
           用户只能靠 epoch loss 突然消失或 acc=nan 来猜测原因。

        3. 系统角度安全:
           空 perm 导致后续 input_id[perm] / node[perm] 均为空 tensor，
           进入采样器时可能触发 CUDA kernel 的 grid_size=0 路径，
           部分 GPU 驱动版本对此未做保护，行为未定义；
           分布式场景下各 rank perm 长度不一致，allreduce shape mismatch
           会直接崩溃 NCCL，且错误信息指向 collective 而非根因。
    """

    def __init__(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError(
                f"[PermDropLastSlicer] batch_size 须 > 0，实际: {batch_size}"
            )
        self.batch_size = batch_size

        if _DBG:
            print(
                f"[WALPURGIS:PermDropLastSlicer.__init__] batch_size={batch_size}",
                file=sys.stderr,
            )

    def apply(self, perm: torch.Tensor, drop_last: bool) -> torch.Tensor:
        """
        对排列张量 perm 执行 drop_last 裁剪。

        Args:
            perm: 1-D 索引排列张量，由 torch.randperm 或 torch.arange 生成。
            drop_last: 是否丢弃无法凑成完整 batch 的尾部样本。

        Returns:
            裁剪后的 perm tensor（d==0 时返回原 tensor，d>0 时返回切片）。
        """
        if not drop_last:
            if _DBG:
                print(
                    f"[WALPURGIS:PermDropLastSlicer.apply] "
                    f"drop_last=False，跳过裁剪，perm.numel()={perm.numel()}",
                    file=sys.stderr,
                )
            return perm

        d = perm.numel() % self.batch_size

        if _DBG:
            print(
                f"[WALPURGIS:PermDropLastSlicer.apply] "
                f"perm.numel()={perm.numel()} batch_size={self.batch_size} "
                f"d={d}",
                file=sys.stderr,
            )

        # 5baac8b 修复核心: d==0 表示整除，无需裁剪；
        # 直接 perm[:-0] 等价于 perm[:0] (空 tensor)，故必须保护
        if d == 0:
            if _DBG:
                print(
                    f"[WALPURGIS:PermDropLastSlicer.apply] "
                    f"d==0 整除，跳过切片，perm 保持完整 numel={perm.numel()}",
                    file=sys.stderr,
                )
            return perm

        # d > 0: 丢弃尾部 d 个多余样本，确保每个 batch 恰好 batch_size 个
        sliced = perm[:-d]

        if _DBG:
            print(
                f"[WALPURGIS:PermDropLastSlicer.apply] "
                f"d={d} > 0，裁剪尾部，裁剪后 perm.numel()={sliced.numel()}",
                file=sys.stderr,
            )

        return sliced
