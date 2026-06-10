"""
dist_tensor.py — 89c9e8d 迁移: 固定CPU内存替代整块复制上GPU

migrate 89c9e8d: [BUG] Pin CPU Memory Instead of Copying to Device

上游变化 (89c9e8d, cugraph-gnn / python/cugraph-pyg/cugraph_pyg/tensor/dist_tensor.py):
  1. DistTensor.__setitem__:
     - 删除: val = val.cuda()  ← 原来把整块 val 先搬到 GPU 显存再 scatter
     - 新增: if not val.is_cuda: val = val.pin_memory()
       ← val 若已在 GPU 则直接 scatter；若在 CPU 则仅锁页(pinned)，
          WholeGraph scatter 底层通过 DMA/UVA 直接从 pinned 内存拉数据，
          无需先把全部 val 占满显存
  2. DistEmbedding.__setitem__:
     - 同上，对称修复（DistEmbedding 各自实现 __setitem__，未复用 DistTensor）
  3. 操作顺序均保持: idx.cuda() → dtype转换 → pin_memory/is_cuda判断 → scatter

Bug 根因:
  - val.cuda() 在 scatter 之前把整个 val tensor 全量搬到 GPU 显存
  - 若 val 比可用显存大（GraphRAG 场景节点特征矩阵动辄数十GB），直接 OOM
  - WholeGraph scatter 本身设计就支持从 pinned host 内存 DMA 写入分布式存储，
    .cuda() 是多余且有害的冗余拷贝，属于设计偏差引入的 critical bug

Walpurgis 改写20%(鲁迅拿法):
  - WalpurgisScatterGuard: 封装 val 的内存状态判断 + pin/noop 决策，
    替代 Python 里两处重复的 if not val.is_cuda: val = val.pin_memory()
  - PinnedValBuffer: 轻量值对象，携带 (val_tensor, was_pinned, was_cuda) 三元状态，
    替代 Python 中散落的局部变量，让调用方可以观测到内存决策路径
  - DistTensorScatter / DistEmbeddingScatter: 把 __setitem__ 逻辑提取为
    可独立测试的静态方法，不依赖完整 WholeGraph 构造
  - 全链路 WALPURGIS_DEBUG=1 断点 print，打印 val 内存状态变化
"""

import os
from typing import Optional

# ──────────────────────────────────────────────
# 调试开关: WALPURGIS_DEBUG=1 开启断点级 print
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS dist_tensor]", *args, **kwargs)


# ──────────────────────────────────────────────
# PinnedValBuffer — val 内存决策的值对象
# ──────────────────────────────────────────────

class PinnedValBuffer:
    """
    记录一次 scatter 前 val tensor 的内存状态决策。

    上游 89c9e8d 的逻辑是:
        if not val.is_cuda:
            val = val.pin_memory()
    我们把这个决策拆成可观测的对象，便于调试和单元测试。

    属性
    ----
    tensor      : 经过 pin/noop 后可直接传入 scatter 的 tensor
    was_cuda    : 原始 val 是否已在 GPU（True → 直接传入，跳过 pin）
    was_pinned  : 原始 val 是否已是 pinned memory（True → pin_memory() 仍会返回新副本，
                  但底层复用已锁页区域，成本极低）
    dtype_cast  : 是否触发了 dtype 转换（val.dtype != target_dtype）
    """

    def __init__(self, tensor, *, was_cuda: bool, was_pinned: bool, dtype_cast: bool):
        self.tensor = tensor
        self.was_cuda = was_cuda
        self.was_pinned = was_pinned
        self.dtype_cast = dtype_cast

    def __repr__(self):
        return (
            f"PinnedValBuffer("
            f"shape={tuple(self.tensor.shape)}, "
            f"dtype={self.tensor.dtype}, "
            f"was_cuda={self.was_cuda}, "
            f"was_pinned={self.was_pinned}, "
            f"dtype_cast={self.dtype_cast})"
        )


# ──────────────────────────────────────────────
# WalpurgisScatterGuard — 核心内存决策逻辑
# ──────────────────────────────────────────────

class WalpurgisScatterGuard:
    """
    封装 89c9e8d 引入的 val 内存策略:
        val.is_cuda  → 直接 scatter（GPU tensor，WholeGraph 可直接读）
        ~val.is_cuda → val.pin_memory() → scatter（锁页后 DMA 拉取，不占完整显存）

    替代上游两处重复的:
        if not val.is_cuda:
            val = val.pin_memory()

    Walpurgis 改写点:
    - 提供 prepare() 静态方法返回 PinnedValBuffer，决策过程可被单独测试
    - dtype 转换在 pin 之前完成（与上游顺序一致: dtype check → pin/noop）
    - WALPURGIS_DEBUG=1 时打印每次决策详情

    上游对应位置:
        DistTensor.__setitem__ 第268行 / DistEmbedding.__setitem__ 第502行
    """

    @staticmethod
    def prepare(val, *, target_dtype) -> "PinnedValBuffer":
        """
        准备 val tensor 供 scatter 使用。

        参数
        ----
        val         : 原始输入 tensor（CPU 或 GPU，pinned 或普通）
        target_dtype: self.dtype，scatter 目标的 dtype

        返回
        ----
        PinnedValBuffer，其 .tensor 可直接传入 WholeGraph scatter
        """
        # ── 断点 1: 进入 guard，记录原始状态 ──────────────────────────
        _dbg(
            f"prepare() 入口: shape={tuple(val.shape)}, dtype={val.dtype}, "
            f"target_dtype={target_dtype}, is_cuda={val.is_cuda}, "
            f"is_pinned={val.is_pinned() if hasattr(val, 'is_pinned') else 'N/A'}"
        )

        was_cuda = val.is_cuda
        was_pinned = val.is_pinned() if hasattr(val, "is_pinned") else False
        dtype_cast = False

        # ── 步骤1: dtype 转换（与上游顺序一致，在 pin 之前）─────────────
        if val.dtype != target_dtype:
            _dbg(f"  dtype 转换: {val.dtype} → {target_dtype}")
            val = val.to(target_dtype)
            dtype_cast = True

        # ── 步骤2: 内存策略决策（89c9e8d 核心修复）───────────────────────
        if not val.is_cuda:
            # CPU tensor: 锁页，让 WholeGraph DMA 直接从 host 拉
            # 89c9e8d 之前是 val.cuda()，把整块搬到 GPU 显存 → OOM
            _dbg(
                f"  val 在 CPU (was_pinned={was_pinned})，"
                f"执行 pin_memory() 而非 .cuda() — 89c9e8d bug fix"
            )
            val = val.pin_memory()
            _dbg(f"  pin_memory() 完成: is_pinned={val.is_pinned()}")
        else:
            # GPU tensor: 直接传入，无需任何操作
            _dbg("  val 已在 GPU，直接传入 scatter，跳过 pin_memory()")

        buf = PinnedValBuffer(
            val,
            was_cuda=was_cuda,
            was_pinned=was_pinned,
            dtype_cast=dtype_cast,
        )

        # ── 断点 2: 决策完成，打印最终状态 ───────────────────────────────
        _dbg(f"prepare() 输出: {buf}")

        return buf


# ──────────────────────────────────────────────
# DistTensorScatter — DistTensor.__setitem__ 的可测逻辑提取
# ──────────────────────────────────────────────

class DistTensorScatter:
    """
    将 DistTensor.__setitem__ 的核心逻辑提取为静态方法。

    上游 89c9e8d 修复后的 DistTensor.__setitem__:
        idx = idx.cuda()
        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        if not val.is_cuda:
            val = val.pin_memory()
        self._tensor.scatter(val, idx)

    Walpurgis 改写: 调用 WalpurgisScatterGuard.prepare() 统一决策，
    逻辑语义完全等价，但决策过程可被观测和单元测试。
    """

    @staticmethod
    def execute(wg_tensor, idx, val, *, dtype):
        """
        执行一次分布式 scatter。

        参数
        ----
        wg_tensor : WholeGraph tensor (已创建)
        idx       : 索引 tensor
        val       : 值 tensor（CPU 或 GPU）
        dtype     : wg_tensor 的目标 dtype

        此调用必须由所有 rank 同步调用（WholeGraph collective 语义）。
        """
        # ── 断点 3: scatter 入口 ──────────────────────────────────────
        _dbg(
            f"DistTensorScatter.execute(): "
            f"idx.shape={tuple(idx.shape)}, val.shape={tuple(val.shape)}"
        )

        # idx 必须在 GPU（与上游一致，idx 始终 .cuda()）
        idx = idx.cuda()
        _dbg(f"  idx → cuda: device={idx.device}")

        # val 内存决策（89c9e8d 核心）
        buf = WalpurgisScatterGuard.prepare(val, target_dtype=dtype)

        # ── 断点 4: 触发 scatter ──────────────────────────────────────
        _dbg(f"  触发 wg_tensor.scatter()，val device={buf.tensor.device}")

        wg_tensor.scatter(buf.tensor, idx)

        _dbg("  scatter 完成")


# ──────────────────────────────────────────────
# DistEmbeddingScatter — DistEmbedding.__setitem__ 的可测逻辑提取
# ──────────────────────────────────────────────

class DistEmbeddingScatter:
    """
    将 DistEmbedding.__setitem__ 的核心逻辑提取为静态方法。

    上游 89c9e8d 修复后的 DistEmbedding.__setitem__:
        idx = idx.cuda()
        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        if not val.is_cuda:
            val = val.pin_memory()
        self._embedding.get_embedding_tensor().scatter(val, idx)

    注意: DistEmbedding 有独立的 __setitem__ 实现（未复用 DistTensor 的），
    89c9e8d 对两个类分别修复，Walpurgis 同样各自提供对应静态方法。
    """

    @staticmethod
    def execute(embedding, idx, val, *, dtype):
        """
        执行一次分布式 embedding scatter。

        参数
        ----
        embedding : WholeGraph WholeMemoryEmbedding 对象
        idx       : 索引 tensor
        val       : 值 tensor（CPU 或 GPU）
        dtype     : embedding 的目标 dtype

        此调用必须由所有 rank 同步调用。
        """
        # ── 断点 5: embedding scatter 入口 ───────────────────────────
        _dbg(
            f"DistEmbeddingScatter.execute(): "
            f"idx.shape={tuple(idx.shape)}, val.shape={tuple(val.shape)}"
        )

        idx = idx.cuda()
        _dbg(f"  idx → cuda: device={idx.device}")

        buf = WalpurgisScatterGuard.prepare(val, target_dtype=dtype)

        # ── 断点 6: 触发 embedding scatter ───────────────────────────
        emb_tensor = embedding.get_embedding_tensor()
        _dbg(
            f"  触发 embedding.get_embedding_tensor().scatter()，"
            f"val device={buf.tensor.device}"
        )

        emb_tensor.scatter(buf.tensor, idx)

        _dbg("  embedding scatter 完成")


# ──────────────────────────────────────────────
# 使用示例（WALPURGIS_DEBUG=1 下可验证决策路径）
# ──────────────────────────────────────────────
#
# class MyDistTensor(DistTensor):
#     def __setitem__(self, idx, val):
#         assert self._tensor is not None, "Please create WholeGraph tensor first."
#         DistTensorScatter.execute(self._tensor, idx, val, dtype=self.dtype)
#
# class MyDistEmbedding(DistEmbedding):
#     def __setitem__(self, idx, val):
#         assert self._tensor is not None, "Please create WholeGraph tensor first."
#         DistEmbeddingScatter.execute(self._embedding, idx, val, dtype=self.dtype)
