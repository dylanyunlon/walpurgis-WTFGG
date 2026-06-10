"""
edge_input_id.py — 66da9ac 迁移: Fix input ID creation to use shape[-1] instead of len

migrate 66da9ac: [BUG] Fix input ID creation to use shape[-1] instead of len

上游变化 (66da9ac):
  文件: python/cugraph-pyg/cugraph_pyg/sampler/distributed_sampler.py
  函数: BaseDistributedSampler.sample_from_edges (L694)

  旧:
      if input_id is None:
          input_id = torch.arange(len(edges), dtype=torch.int64, device="cpu")

  新:
      if input_id is None:
          input_id = torch.arange(edges.shape[-1], dtype=torch.int64, device="cpu")

  edges 是 2 x N 的 tensor（文档注释: "2 x (# edges) tensor of edges to sample from"）。
  len(edges) 返回第一维度 → 常数 2，而非边数 N。
  edges.shape[-1] 取最后一维 → 正确得到 N。

  此 bug 导致 input_id 恒为 [0, 1]，无论实际边数多少。
  任何超过 2 条边的 sample_from_edges 调用，input_id 与实际 batch 数量严重不符，
  下游 batch 索引越界或截断，采样数据静默损坏。

Knuth 审查:
  1. diff 对比源:
     | 上游 66da9ac                              | Walpurgis 迁移                            |
     |---|---|
     | `torch.arange(edges.shape[-1], ...)`     | `make_edge_input_id(edges)` 封装函数      |
     | 无前置断言                                 | `validate_edges_shape` 检查 edges 是 2×N  |
     | 无调试输出                                 | WALPURGIS_DEBUG=1 打印 len vs shape[-1]   |
     | 一行修复，无命名                            | `EdgeInputIdError` 异常类，错误信息完整     |

  2. 用户角度 bug:
     - 调用 sampler.sample_from_edges(edges) 时，不传 input_id（最常见用法）。
       edges 有 1024 条，但 input_id 被创建为 tensor([0, 1])，长度为 2。
       下游 __get_call_groups 以 input_id 长度为准切分 batch，
       导致只有 2 个 seed 被采样，其余 1022 条边静默丢失。
       训练 loss 异常低（数据量严重不足），但无任何异常抛出，难以定位。

  3. 系统角度安全:
     - len() 对 PyTorch tensor 返回第 0 维长度，这是 Python 数据模型的标准行为，
       与 numpy ndarray 一致。但边索引约定是 2×N，第 0 维永远是 2（src/dst 行）。
       此处 len() 的"正确"语义（返回行数）与业务语义（返回边数）完全相反，
       是 API 语义陷阱（semantic trap），不是代码拼写错误。
     - 静默截断：input_id 短于 edges.shape[-1] 时，上游代码不做长度一致性校验，
       短 input_id 被正常传入 __get_call_groups，batch 分组以最短 tensor 为准，
       多余的边被无声丢弃。系统不崩溃，训练不报错，用户观察不到。
     - 只有 edges 恰好只有 2 列（即 N=2）时旧代码才偶然正确，
       此特殊情况可能掩盖 bug，导致小规模测试通过但生产规模失效。

Walpurgis 改写 20%（鲁迅拿法）:
  - `validate_edges_shape`: 封装 edges 维度校验，检查 ndim==2 且 shape[0]==2，
    上游无此前置检查，非法形状下 shape[-1] 也能返回但语义错误
  - `make_edge_input_id`: 替代内联 torch.arange，统一 input_id 构建入口，
    断点调试打印 len(edges) vs edges.shape[-1]（核心 bug 两侧对比）
  - `EdgeInputIdError`: 专用异常，携带 edges 形状信息，上游直接 IndexError 或静默
  - `assert_input_id_consistent`: 事后一致性校验，edges.shape[-1] == len(input_id)，
    与 make_edge_input_id 配套使用，防止调用方传入错误长度的自定义 input_id

作者: dylanyunlon<dogechat@163.com>
"""
import sys
import os
from typing import Optional

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_eid(tag: str, msg: str) -> None:
    """断点调试: edge_input_id 专用 print"""
    if _DBG:
        print(f"[DEBUG 66da9ac {tag}] {msg}", file=sys.stderr, flush=True)


# ─── EdgeInputIdError: 上游无此异常类型 ──────────────────────────────────────
# Knuth审查 系统角度: 静默截断不抛异常，需要显式错误类
class EdgeInputIdError(ValueError):
    """
    Raised when edges tensor has unexpected shape for input_id construction.

    上游 66da9ac 只修复了 arange 参数，未对非法形状做保护。
    改写：形状校验失败时抛此异常，携带实际 shape 信息。
    """
    pass


# ─── validate_edges_shape: 上游无此校验 ──────────────────────────────────────
# Python (66da9ac distributed_sampler.py):
#   edges = torch.as_tensor(edges, device="cuda")   # 转换后形状未校验
#   ...
#   input_id = torch.arange(edges.shape[-1], ...)   # 直接取 shape[-1]
#
# 改写: 在 make_edge_input_id 调用前做 ndim/shape[0] 前置校验
def validate_edges_shape(edges) -> None:
    """
    校验 edges tensor 形状符合 2×N 约定。

    上游 distributed_sampler.py 文档: "2 x (# edges) tensor of edges to sample from."
    此约定在代码中无前置检查；传入转置的 N×2 tensor 时，
    旧代码 len(edges) 返回 N（偶然正确），新代码 edges.shape[-1] 返回 2（错误）。
    故 validate_edges_shape 同时保护旧行为和新行为下的正确性。

    断点调试: 打印 edges.shape, len(edges), edges.shape[-1] 三者对比
    """
    # 延迟导入 torch，避免模块级强依赖
    try:
        import torch
        is_tensor = isinstance(edges, torch.Tensor)
    except ImportError:
        is_tensor = False

    try:
        import numpy as np
        is_ndarray = isinstance(edges, np.ndarray)
    except ImportError:
        is_ndarray = False

    if not (is_tensor or is_ndarray):
        # 鸭子类型：只要有 .shape 属性就继续
        if not hasattr(edges, 'shape'):
            _dbg_eid(
                "validate_edges_shape",
                f"edges has no .shape attribute: type={type(edges).__name__!r}"
            )
            raise EdgeInputIdError(
                f"edges must have a .shape attribute (torch.Tensor or numpy.ndarray). "
                f"Got {type(edges).__name__!r}."
            )

    shape = edges.shape
    ndim = len(shape)

    # ── 断点调试核心: 打印 len(edges) vs shape[-1] 对比，直指 bug 根因 ──
    _dbg_eid(
        "validate_edges_shape",
        f"edges.shape={shape} | "
        f"len(edges)={len(edges)} [旧: 恒为 {shape[0] if ndim > 0 else '?'}，通常=2] | "
        f"edges.shape[-1]={shape[-1] if ndim > 0 else '?'} [新: 边数 N，正确]"
    )

    if ndim != 2:
        raise EdgeInputIdError(
            f"edges must be a 2D tensor of shape (2, N). "
            f"Got ndim={ndim}, shape={shape}. "
            f"(66da9ac: len(edges) 在 ndim!=2 时行为更加不可预期)"
        )

    if shape[0] != 2:
        raise EdgeInputIdError(
            f"edges must have shape[0]==2 (src/dst rows). "
            f"Got shape={shape}. "
            f"Possible transposed input (N×2 instead of 2×N)? "
            f"(Knuth审查: 转置输入下旧代码 len(edges) 偶然正确=N，"
            f"新代码 shape[-1] 返回 2，两者都隐蔽)"
        )

    _dbg_eid("validate_edges_shape", f"OK: shape={shape}, num_edges={shape[-1]}")


# ─── make_edge_input_id: 替代内联 torch.arange(edges.shape[-1], ...) ─────────
# Python (66da9ac distributed_sampler.py:693-694):
#
#   旧 (BUG):
#       if input_id is None:
#           input_id = torch.arange(len(edges), dtype=torch.int64, device="cpu")
#                                    ^^^^^^^^^^
#                                    len(edges) = 2（第0维），恒为2，与边数无关
#
#   新 (FIX):
#       if input_id is None:
#           input_id = torch.arange(edges.shape[-1], dtype=torch.int64, device="cpu")
#                                    ^^^^^^^^^^^^^^^^
#                                    shape[-1] = N（第1维），正确的边数
#
# 改写: 封装为函数，加 validate_edges_shape 前置校验，加断点调试
def make_edge_input_id(edges, dtype=None, device="cpu"):
    """
    构造边采样的 input_id tensor，对应 66da9ac 修复后的逻辑。

    Parameters
    ----------
    edges : torch.Tensor
        2 x N 的边 tensor（src/dst 两行，N 列对应 N 条边）。
    dtype : torch.dtype, optional
        output dtype，默认 torch.int64（与上游一致）。
    device : str
        output device，默认 "cpu"（与上游一致）。

    Returns
    -------
    torch.Tensor
        torch.arange(N, dtype=int64, device=cpu)，长度 = 边数 N。

    断点调试 (WALPURGIS_DEBUG=1):
        打印 len(edges) [旧值=2] vs edges.shape[-1] [新值=N] 两侧对比，
        直接显示 bug 根因。

    Usage:
        edges = torch.randint(0, 100, (2, 512), device="cuda")
        input_id = make_edge_input_id(edges)   # tensor([0, 1, ..., 511])
        # 旧代码: torch.arange(len(edges)) → tensor([0, 1])  ← 严重错误
        # 新代码: torch.arange(edges.shape[-1]) → tensor([0, ..., 511])  ← 正确
    """
    import torch

    if dtype is None:
        dtype = torch.int64

    # 前置校验（上游无此步骤）
    validate_edges_shape(edges)

    num_edges_wrong = len(edges)      # 旧: 恒为 2（第0维=src/dst行数）
    num_edges_right = edges.shape[-1]  # 新: N（第1维=实际边数）

    # ── 断点调试：核心对比，一眼看出 bug ──
    _dbg_eid(
        "make_edge_input_id",
        f"edges.shape={edges.shape} | "
        f"len(edges)={num_edges_wrong} [旧BUG: input_id长度={num_edges_wrong}，"
        f"与实际边数无关] | "
        f"edges.shape[-1]={num_edges_right} [新FIX: input_id长度={num_edges_right}，正确]"
    )

    if num_edges_wrong != num_edges_right:
        # 最典型情形：edges.shape=(2, N>2)，旧代码 arange(2) 而非 arange(N)
        _dbg_eid(
            "make_edge_input_id",
            f"BUG WOULD HAVE OCCURRED: 旧代码产生 input_id=[0,1] (len=2)，"
            f"实际需要 input_id=[0,...,{num_edges_right-1}] (len={num_edges_right})。"
            f"丢失 {num_edges_right - num_edges_wrong} 条边的 input_id。"
        )
    else:
        # N==2 时 len(edges)==shape[-1]，旧代码偶然正确
        _dbg_eid(
            "make_edge_input_id",
            f"NOTE: len(edges)==edges.shape[-1]=={num_edges_right}，"
            f"旧代码在此恰好正确（N=2 特殊情况，可能掩盖 bug）"
        )

    input_id = torch.arange(num_edges_right, dtype=dtype, device=device)

    _dbg_eid(
        "make_edge_input_id",
        f"input_id created: shape={input_id.shape} dtype={input_id.dtype} device={input_id.device}"
    )

    return input_id


# ─── assert_input_id_consistent: 上游无此后置校验 ──────────────────────────
# Knuth审查 用户角度: 用户传入自定义 input_id 时可能长度不符，上游无校验
def assert_input_id_consistent(edges, input_id) -> None:
    """
    后置一致性校验：input_id 长度必须等于 edges.shape[-1]。

    对应 66da9ac 修复的反面：即使 input_id 由外部传入而非自动构造，
    也应确保长度与边数一致。上游 sample_from_edges 无此检查。

    断点调试: 打印 edges.shape[-1] vs len(input_id)

    Usage:
        assert_input_id_consistent(edges, input_id)  # 在 sample_from_edges 入口调用
    """
    num_edges = edges.shape[-1]
    num_ids = len(input_id)

    _dbg_eid(
        "assert_input_id_consistent",
        f"edges.shape[-1]={num_edges} | len(input_id)={num_ids}"
    )

    if num_edges != num_ids:
        _dbg_eid(
            "assert_input_id_consistent",
            f"MISMATCH: edges has {num_edges} cols but input_id has {num_ids} elements"
        )
        raise EdgeInputIdError(
            f"input_id length mismatch: edges.shape[-1]={num_edges} "
            f"but len(input_id)={num_ids}. "
            f"(66da9ac: 旧代码 len(edges) 创建长度=2 的 input_id，"
            f"此校验可捕获类似维度混淆错误)"
        )

    _dbg_eid("assert_input_id_consistent", f"OK: both have {num_edges} elements")


# ─── resolve_edge_input_id: 对应 sample_from_edges 入口逻辑完整封装 ─────────
# Python (66da9ac distributed_sampler.py:693-694):
#   if input_id is None:
#       input_id = torch.arange(edges.shape[-1], dtype=torch.int64, device="cpu")
#
# 改写: 将 None-check + 构造 + 一致性校验 三步合并为单函数
def resolve_edge_input_id(edges, input_id=None, dtype=None, device="cpu"):
    """
    对应 sample_from_edges 中 input_id 解析逻辑的完整封装（含 66da9ac 修复）。

    旧行为 (BUG):
        if input_id is None:
            input_id = torch.arange(len(edges), ...)  # len=2, 错误

    新行为 (FIX + 改写):
        if input_id is None:
            input_id = make_edge_input_id(edges)      # shape[-1], 正确
        else:
            assert_input_id_consistent(edges, input_id)  # 校验传入的 id

    Parameters
    ----------
    edges : torch.Tensor
        2×N 边 tensor。
    input_id : torch.Tensor or None
        外部传入的 input_id，None 时自动构造。
    dtype : torch.dtype, optional
        仅在 input_id=None 时有效。
    device : str
        仅在 input_id=None 时有效。

    Returns
    -------
    torch.Tensor
        长度为 N 的 input_id tensor。

    Usage (drop-in 替代 sample_from_edges 中的 if input_id is None 块):
        input_id = resolve_edge_input_id(edges, input_id)
    """
    _dbg_eid(
        "resolve_edge_input_id",
        f"input_id={'None (auto-construct)' if input_id is None else f'provided (len={len(input_id)})'}"
    )

    if input_id is None:
        # 66da9ac 修复路径: 使用 edges.shape[-1] 而非 len(edges)
        return make_edge_input_id(edges, dtype=dtype, device=device)
    else:
        # 用户传入路径: 后置一致性校验（上游无此步骤）
        import torch
        input_id = torch.as_tensor(input_id, device=device)
        assert_input_id_consistent(edges, input_id)
        _dbg_eid(
            "resolve_edge_input_id",
            f"using provided input_id: shape={input_id.shape}"
        )
        return input_id
