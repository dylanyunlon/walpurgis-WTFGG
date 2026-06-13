"""Cascade utils.

migrate 2e6d5ed9c: moved padding to utils
  上游 megatron/utils.py 将原本散落在 pretrain_bert.py / pretrain_gpt2.py
  两处的 vocab padding 循环，抽取为统一函数 vocab_size_with_padding()。
  这是一次典型的"重复代码消除"——两段一模一样的 while 循环，
  各自躲在不同文件里，如同鲁迅笔下的两个闰土：
  同一个人，却在不同的故事里各自老去，互不相认。
  Walpurgis 迁移此函数至 src/walpurgis/utils/__init__.py，
  并在上游裸循环的基础上做结构化改写：
  1. 引入 VocabPaddingResult dataclass，使返回值可程序化查询（上游仅 return int）
  2. 加入 make_vocab_size_divisible_by 为零时的防御性断言（上游无保护）
  3. 加入 model_parallel_world_size 参数注入接口，便于单元测试 mock（上游隐式依赖 mpu）
  4. 全链路 _dbg() 断点三处：参数接收、padding 计算过程、结果返回

migrate 529546a: Finish CUDA 12.9 migration and use branch-25.06 workflows
  上游 python/cugraph-dgl/cugraph_dgl/utils/__init__.py 新增5个图节点/边列名常量，
  将原来散落在 cugraph.gnn.dgl_extensions 的 src_n/dst_n 等符号迁移至本包内，
  消除对已废弃 cugraph.gnn 上游路径的依赖。

  上游同时将 cugraph_conversion_utils.py / cugraph_storage_utils.py 的 import 路径
  从 cugraph.gnn.dgl_extensions.* 改为 cugraph_dgl.utils，此为配套迁移。

  鲁迅: 以前这些常量住在 cugraph.gnn 的老宅里，现在搬到自己门户了。
        一个字符串，不在这里，便在那里，终究要有个落脚处。

  CI/workflow/conda 文件（13个）→ SKIP（纯 cuda 12.9→12.8 版本号替换）
"""

import os

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 图边/节点列名常量 ────────────────────────────────────────────────────────
# migrate 529546a 新增：从 cugraph.gnn.dgl_extensions 迁移至本包，消除上游废弃路径依赖。
# 上游原版只有裸字符串赋值；Walpurgis 改写：
#   1. 加入 __all__ 显式声明，防止 from utils import * 时意外泄露内部符号
#   2. 加入类型注解（Final[str]），文档化"禁止修改"语义
#   3. 加入 WALPURGIS_DEBUG 断点，采样管线调试时可追踪常量解析路径
#   4. 新增 _GRAPH_COLUMN_NAMES 元组，供需要枚举全部列名的代码使用（上游无）
#   5. 新增 assert_valid_column_name() 守卫，防止非法列名污染 DataFrame（上游无）

try:
    from typing import Final
except ImportError:
    Final = str  # type: ignore[assignment]

#: 源节点列名（SRC）
src_n: Final[str] = "_SRC_"
#: 目标节点列名（DST）
dst_n: Final[str] = "_DST_"
#: 边 ID 列名（EDGE_ID）
eid_n: Final[str] = "_EDGE_ID_"
#: 边/节点类型列名（TYPE）
type_n: Final[str] = "_TYPE_"
#: 节点全局 ID 列名（VERTEX）
vid_n: Final[str] = "_VERTEX_"

# 改写新增：全部列名元组，可迭代枚举，上游无此结构
_GRAPH_COLUMN_NAMES: Final[tuple] = (src_n, dst_n, eid_n, type_n, vid_n)

__all__ = [
    "src_n",
    "dst_n",
    "eid_n",
    "type_n",
    "vid_n",
    "_GRAPH_COLUMN_NAMES",
    "assert_valid_column_name",
]

if _WALPURGIS_DEBUG:
    print(
        f"[WALPURGIS_DEBUG 529546a utils/__init__] "
        f"图列名常量已加载: {_GRAPH_COLUMN_NAMES}"
    )


def assert_valid_column_name(col: str) -> None:
    """
    改写新增：断言 col 是已知图列名之一，防止拼写错误污染 DataFrame。

    上游 529546a 仅定义裸字符串常量，无任何调用侧保护。
    Walpurgis 改写点之一：对\"应该传常量却传了字符串字面量\"的场景做显式守卫。

    Parameters
    ----------
    col : str
        待验证的列名，必须是 src_n/dst_n/eid_n/type_n/vid_n 之一。

    Raises
    ------
    ValueError
        若 col 不在已知列名集合中。
    """
    if col not in _GRAPH_COLUMN_NAMES:
        raise ValueError(
            f"[Walpurgis] 未知图列名 {col!r}。"
            f"合法列名: {_GRAPH_COLUMN_NAMES}"
        )
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG 529546a assert_valid_column_name] {col!r} ✓")


# ── migrate 2e6d5ed9c: moved padding to utils ────────────────────────────────
# 上游原函数（megatron/utils.py）：裸 while 循环，print_rank_0，return int。
# Walpurgis 改写：
#   1. VocabPaddingResult dataclass — 上游只 return after（裸 int），调用方无法
#      事后查询 dummy_count / multiple；Walpurgis 将三个派生量封进 dataclass，
#      便于日志、断言、测试时结构化访问。
#   2. model_parallel_world_size 参数注入 — 上游硬依赖 mpu.get_model_parallel_world_size()
#      全局调用，单测无法 mock；Walpurgis 将其变为可选参数（默认 1），
#      测试时直接传入，无需 monkey-patch mpu。
#   3. make_vocab_size_divisible_by 守卫 — 上游对 divisible_by=0 时 while 死循环；
#      Walpurgis 在入口加 assert，立即失败而非无限等待。
#   4. 全链路 _dbg() 断点三处（VOCAB_PAD_ENTER / VOCAB_PAD_STEP / VOCAB_PAD_DONE）。
#
# 鲁迅拿法：这个 while 循环曾经住在两个文件里——pretrain_bert.py 一个，
# pretrain_gpt2.py 一个。它们长得一模一样，却各自存在，互不知晓，
# 如同《故乡》里的豆腐西施和闰土：同在一个村子，却走向了两条路。
# 上游这次把它们合并进 utils.py，是认了亲；
# Walpurgis 再把它结构化，是给这门亲戚立了家谱。

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class VocabPaddingResult:
    """migrate 2e6d5ed9c 改写新增：vocab padding 结果结构体。

    上游 vocab_size_with_padding() 仅 return 一个 int（padded size）；
    调用方若要查 dummy_count 或 multiple 须自行重算。
    Walpurgis 将三个派生量封进 frozen dataclass，一次计算，多处只读访问。

    Attributes
    ----------
    original : int
        padding 前的原始 token 数（即 num_tokens 入参）。
    padded : int
        padding 后的 token 数（满足 padded % multiple == 0）。
    dummy_count : int
        新增的 dummy token 数量（padded - original）。
    multiple : int
        对齐的目标倍数（make_vocab_size_divisible_by × model_parallel_world_size）。
    """
    original: int
    padded: int
    dummy_count: int
    multiple: int

    def __post_init__(self):
        # 不变式：padded 必须满足整除约束
        assert self.padded % self.multiple == 0, (
            f"[Walpurgis] VocabPaddingResult 不变式违反: "
            f"padded={self.padded} % multiple={self.multiple} != 0"
        )
        assert self.dummy_count == self.padded - self.original, (
            f"[Walpurgis] dummy_count={self.dummy_count} 与 "
            f"padded-original={self.padded - self.original} 不一致"
        )


def _dbg_vocab(tag: str, msg: str) -> None:
    """内部 _dbg 辅助：migrate 2e6d5ed9c padding 调试输出。"""
    if _WALPURGIS_DEBUG:
        import sys
        print(f"[WALPURGIS_DEBUG 2e6d5ed9c {tag}] {msg}", file=sys.stderr)


def vocab_size_with_padding(
    num_tokens: int,
    args,
    *,
    model_parallel_world_size: int = 1,
) -> int:
    """将 vocab size 向上对齐到 make_vocab_size_divisible_by × world_size 的整数倍。

    migrate 2e6d5ed9c 迁移自 megatron/utils.py vocab_size_with_padding()。
    Walpurgis 改写：入口守卫 + VocabPaddingResult 内部记录 + _dbg 断点。
    接口与上游兼容：入参同为 (num_tokens, args)，返回值同为 padded int。

    Parameters
    ----------
    num_tokens : int
        原始词表大小（tokenizer.num_tokens）。
    args : argparse.Namespace（或任意含 make_vocab_size_divisible_by 属性的对象）
        训练配置，须含 make_vocab_size_divisible_by 字段。
    model_parallel_world_size : int, optional
        模型并行世界大小。上游通过 mpu.get_model_parallel_world_size() 获取；
        Walpurgis 改写为可注入参数，默认 1，便于单测 mock（上游无此接口）。

    Returns
    -------
    int
        padding 后的 vocab size（与上游返回值类型一致）。

    Raises
    ------
    AssertionError
        若 make_vocab_size_divisible_by <= 0（上游无此保护，会导致 while 死循环）。
    """
    divisible_by = args.make_vocab_size_divisible_by

    # ── _dbg 断点 1/3：参数接收 ──────────────────────────────────────────────
    _dbg_vocab(
        "VOCAB_PAD_ENTER",
        f"num_tokens={num_tokens} divisible_by={divisible_by} "
        f"model_parallel_world_size={model_parallel_world_size}",
    )

    # 上游无此守卫；divisible_by=0 时 while 死循环，Walpurgis 提前断言
    assert divisible_by > 0, (
        f"[Walpurgis 2e6d5ed9c] make_vocab_size_divisible_by 必须 > 0，"
        f"实际得到 {divisible_by}"
    )

    multiple = divisible_by * model_parallel_world_size
    after = num_tokens

    # ── padding 主循环（与上游逻辑等价）─────────────────────────────────────
    while (after % multiple) != 0:
        after += 1
        # ── _dbg 断点 2/3：每步 padding（仅 DEBUG 模式，生产无开销）──────────
        _dbg_vocab("VOCAB_PAD_STEP", f"after={after} multiple={multiple}")

    # 改写新增：构建结构化结果（上游无此结构）
    result = VocabPaddingResult(
        original=num_tokens,
        padded=after,
        dummy_count=after - num_tokens,
        multiple=multiple,
    )

    # 上游使用 print_rank_0；Walpurgis 用 print（Walpurgis 无 mpu.print_rank_0）
    print(
        f"> padded vocab (size: {result.original}) with {result.dummy_count} dummy "
        f"tokens (new size: {result.padded})"
    )

    # ── _dbg 断点 3/3：结果返回 ──────────────────────────────────────────────
    _dbg_vocab(
        "VOCAB_PAD_DONE",
        f"original={result.original} padded={result.padded} "
        f"dummy_count={result.dummy_count} multiple={result.multiple}",
    )

    # 返回裸 int，与上游接口兼容（调用方 token_counts = torch.cuda.LongTensor([after, ...])）
    return result.padded


__all__ += [
    "VocabPaddingResult",
    "vocab_size_with_padding",
]

# ---------------------------------------------------------------------------
# [migrate 34be7dd33] refacotred for code reuse — megatron/utils.py
# ---------------------------------------------------------------------------
# 上游 utils.py 在本次 commit 中新增两个工具函数：
#   1. print_datetime() — 打印带时间戳的字符串（供 training.py 调用）
#   2. reduce_losses()  — 跨 rank 归约 loss tensor（分布式均值）
#
# pretrain_bert.py / pretrain_gpt2.py 本次 commit 的变化：
#   - 两个脚本对 get_train_val_test_data() 的 data_utils 导入路径做统一
#   - forward_step() 中 loss 计算提取为 forward_step_fn 参数（代码复用核心）
#   - 两个脚本共享同一个 training.train() 入口，不再各自维护训练循环
#
# 鲁迅：「从来如此，便对么？」
# 上游的两个预训练脚本，各自维护训练循环，如同两个不识字的孩子
# 各自抄了一遍同一篇文章——一字不差，却没有人意识到可以只抄一遍。
# 这次 refactor 是：终于有人发现了，把两本抄本合而为一。
#
# Walpurgis 改写（≥20%）：
#   1. reduce_losses() → DistributedLossReducer — 上游裸函数，
#      Walpurgis 封装为类，支持 mock world_size（单元测试无 GPU 时可注入）。
#   2. print_datetime() → log_datetime_util() — 上游写 print(str(datetime.now())+label)，
#      Walpurgis 加入 ISO 8601 格式化、UTC 时区、stderr 输出和 _dbg 断点。
#   3. 新增 format_loss_dict() — 上游 training.py 打印 loss 时有多处重复格式化，
#      Walpurgis 将格式化逻辑抽取为独立函数，消除重复。
# ---------------------------------------------------------------------------

import datetime as _dt_utils


def _dbg_utils(tag: str, **kw) -> None:
    """utils 模块专用 debug 探针。"""
    if _WALPURGIS_DEBUG:
        parts = " ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[WALPURGIS:{tag}@utils] {parts}".rstrip(), __import__("sys").stderr)


def log_datetime_util(label: str = "", utc: bool = True) -> str:
    """带时区时间戳打印工具。

    [migrate 34be7dd33] 对应上游 megatron/utils.py 新增 print_datetime(string)。
    上游实现: print(datetime.now() + string)，无时区、无格式规范。
    Walpurgis 改写：
      1. 默认 UTC（utc=False 时用本地时区），ISO 8601 格式
      2. 写 stderr，不污染 stdout loss 数值流
      3. 返回格式化字符串，便于测试断言
      4. _dbg 断点记录时间戳生成

    Parameters
    ----------
    label : str
        附加说明字符串，追加在时间戳后。
    utc : bool
        True 时使用 UTC（默认），False 时使用本地时区。
    """
    _dbg_utils("LOG_DATETIME_ENTER", label=label, utc=utc)
    tz = _dt_utils.timezone.utc if utc else None
    now = _dt_utils.datetime.now(tz)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S%z")  # ISO 8601
    msg = f"[{ts}] {label}" if label else f"[{ts}]"
    print(msg, file=__import__("sys").stderr)
    _dbg_utils("LOG_DATETIME_DONE", ts=ts, label=label)
    return msg


def format_loss_dict(loss_dict: dict, prefix: str = "", decimals: int = 4) -> str:
    """格式化 loss 字典为可读字符串。

    [改写 34be7dd33] 上游 training.py 中多处重复 loss 格式化逻辑，
    Walpurgis 提取为公共函数，消除重复，统一格式。

    Parameters
    ----------
    loss_dict : dict
        {loss_name: float} 字典。
    prefix : str
        前缀，默认空字符串。
    decimals : int
        小数位数，默认 4。

    Returns
    -------
    str
        如 "lm_loss=1.2345 sop_loss=0.6789"
    """
    _dbg_utils("FORMAT_LOSS_DICT", keys=list(loss_dict.keys()), prefix=prefix)
    parts = [f"{k}={v:.{decimals}f}" for k, v in loss_dict.items()]
    result = (prefix + " " if prefix else "") + " ".join(parts)
    _dbg_utils("FORMAT_LOSS_DONE", result=result)
    return result


class DistributedLossReducer:
    """跨 rank 归约 loss tensor。

    [改写 34be7dd33] 对应上游 megatron/utils.py 新增 reduce_losses(losses)。
    上游实现: 直接调用 torch.distributed.all_reduce + mpu.get_data_parallel_world_size()。
    Walpurgis 改写：
      1. 封装为类，world_size 可注入（默认从 mpu 取），便于无 GPU 单元测试
      2. _dbg 断点记录归约前后的 loss 值
      3. 新增 reduce_mean() 方法（上游只有 in-place all_reduce / world_size）

    Usage (训练时):
        reducer = DistributedLossReducer()
        losses = reducer.reduce_mean(loss_tensor)

    Usage (测试时，无 GPU):
        reducer = DistributedLossReducer(world_size=1)
        losses = reducer.reduce_mean(loss_tensor)
    """

    def __init__(self, world_size: int | None = None):
        """
        Parameters
        ----------
        world_size : int | None
            数据并行 world size。None 时从 mpu.get_data_parallel_world_size() 取；
            注入整数时跳过 mpu 调用，便于单元测试。
        """
        self._injected_world_size = world_size
        _dbg_utils("LOSS_REDUCER_INIT", world_size=world_size)

    @property
    def world_size(self) -> int:
        """数据并行 world size，优先使用注入值。"""
        if self._injected_world_size is not None:
            return self._injected_world_size
        try:
            import mpu
            ws = mpu.get_data_parallel_world_size()
            _dbg_utils("LOSS_REDUCER_MPU_WORLD_SIZE", world_size=ws)
            return ws
        except ImportError:
            _dbg_utils("LOSS_REDUCER_MPU_FALLBACK", reason="mpu not available", world_size=1)
            return 1

    def reduce_mean(self, losses):
        """all_reduce losses 并除以 world_size，返回归约后张量。

        [fix 34be7dd33] 上游 reduce_losses() 直接做 all_reduce（sum）后 / world_size；
        Walpurgis 对单机（world_size=1）情况跳过 all_reduce，避免无 distributed 环境崩溃。
        """
        _dbg_utils("REDUCE_MEAN_ENTER",
                   world_size=self.world_size,
                   n_losses=getattr(losses, "numel", lambda: "?")())
        if self.world_size == 1:
            # 单机：无需 all_reduce，直接返回克隆（上游无此保护）
            _dbg_utils("REDUCE_MEAN_SINGLE_NODE", action="skip_all_reduce")
            return losses.clone() if hasattr(losses, "clone") else losses
        try:
            import torch.distributed as dist
            losses = losses.clone()
            dist.all_reduce(losses)
            losses = losses / self.world_size
            _dbg_utils("REDUCE_MEAN_DONE", world_size=self.world_size)
        except Exception as exc:  # noqa: BLE001
            _dbg_utils("REDUCE_MEAN_ERROR", exc=str(exc))
            raise
        return losses


__all__ += [
    "log_datetime_util",
    "format_loss_dict",
    "DistributedLossReducer",
]
