"""Walpurgis 词表对齐工具：将词表大小填充至模型并行整除倍数。

[migrate ebbe40cd3] Merge branch 'move_vocab_padding_to_utils' into 'master'
  上游: megatron/utils.py 新增 vocab_size_with_padding()，
  同时将 pretrain_bert.py / pretrain_gpt2.py 中各自重复的内联填充逻辑统一抽走。
  上游实现: 裸 while 循环逐一递增 after，直到整除
  make_vocab_size_divisible_by × model_parallel_world_size，
  配合 print_rank_0 打印摘要。

  鲁迅所言:「同样的话，在两个地方说了两遍，第三遍就只剩空话了。」
  上游两处 pretrain 脚本各抄了一份同样的 padding 循环，
  像是写信时把同一段嘱咐誊了两遍，誊完还不确定哪份才算数。
  Walpurgis 将此逻辑收拢为 VocabPaddingPolicy，职责明确三分：
    1. _compute_multiple: 计算实际对齐模数（divisible_by × world_size），
       _dbg 断点可见世界规模如何参与计算；
    2. compute: 纯函数式计算填充后词表大小，无副作用，可单独测试；
    3. apply: 计算 + 打印摘要，对应上游 vocab_size_with_padding() 完整行为。
  上游 print_rank_0 在 Walpurgis 中以可注入的 rank_print_fn 替代，
  默认行为等价，但测试时可传入 lambda: None 静默。
"""

import os
import sys
from typing import Callable, Optional

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"
_CAS_DBG = os.environ.get("CASCADE_DEBUG", "0") == "1"


def _dbg(tag: str, **kw) -> None:
    """Unified debug probe; fires when WALPURGIS_DEBUG=1 or CASCADE_DEBUG=1."""
    if _WDBG or _CAS_DBG:
        parts = " ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[WALPURGIS:{tag}@vocab_padding] {parts}".rstrip(), file=sys.stderr)


# ---------------------------------------------------------------------------
# [migrate ebbe40cd3] vocab_size_with_padding — 上游原版
# ---------------------------------------------------------------------------
# 上游 megatron/utils.py 2e6d5ed9c 新增：
#
#   def vocab_size_with_padding(num_tokens, args):
#       after = num_tokens
#       multiple = args.make_vocab_size_divisible_by * \
#                  mpu.get_model_parallel_world_size()
#       while (after % multiple) != 0:
#           after += 1
#       print_rank_0('> padded vocab (size: {}) with {} dummy '
#                    'tokens (new size: {})'.format(
#                        num_tokens, after - num_tokens, after))
#       return after
#
# 上游两处调用点：
#   pretrain_bert.py  — 替换内联 before/after/multiple/while 循环
#   pretrain_gpt2.py  — 同上
#
# 鲁迅: 「同样的话，在两个地方说了两遍，第三遍就只剩空话了。」
# 上游的解法是把循环剪出来放进 utils，再在两处 import，干净利落但无审计线索。
# Walpurgis 将其封装为 VocabPaddingPolicy：
#   - _compute_multiple: 计算对齐模数，world_size 通过依赖注入而非全局 mpu 调用
#   - compute: 纯函数计算填充后大小，无副作用，可在无 GPU 环境单测
#   - apply: 完整流程（计算 + 摘要打印），等价于上游 vocab_size_with_padding()
# ---------------------------------------------------------------------------


class VocabPaddingPolicy:
    """词表填充策略：计算并应用模型并行对齐填充。

    [migrate ebbe40cd3] 对应上游 megatron/utils.py vocab_size_with_padding()。
    Walpurgis 改写：将 mpu.get_model_parallel_world_size() 全局调用改为
    可注入的 world_size_fn，解耦 mpu 依赖，使策略可在无分布式环境下实例化与测试。
    上游无此解耦，直接在函数体内调 mpu 全局函数。
    """

    def __init__(
        self,
        make_vocab_size_divisible_by: int,
        world_size_fn: Callable[[], int],
        rank_print_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Parameters
        ----------
        make_vocab_size_divisible_by : int
            对应 args.make_vocab_size_divisible_by。
            上游直接从 args 读取；Walpurgis 显式传入，依赖清晰。
        world_size_fn : Callable[[], int]
            返回模型并行世界规模，对应上游 mpu.get_model_parallel_world_size()。
            注入式设计：测试可传 lambda: 1，生产传 mpu.get_model_parallel_world_size。
        rank_print_fn : Callable[[str], None], optional
            打印函数，对应上游 print_rank_0；默认 print，测试可传 lambda _: None。
        """
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self._world_size_fn = world_size_fn
        self._rank_print_fn: Callable[[str], None] = (
            rank_print_fn if rank_print_fn is not None else print
        )
        _dbg(
            "POLICY_INIT",
            divisible_by=make_vocab_size_divisible_by,
            rank_print_fn=getattr(rank_print_fn, "__name__", repr(rank_print_fn)),
        )

    def _compute_multiple(self) -> int:
        """计算实际对齐模数 = divisible_by × world_size。

        [migrate ebbe40cd3] 对应上游:
            multiple = args.make_vocab_size_divisible_by * \\
                       mpu.get_model_parallel_world_size()
        Walpurgis 将此抽为独立方法，world_size 延迟求值（调用时而非构造时），
        与上游语义等价，但便于 _dbg 断点观察 world_size 参与计算的时机。
        """
        world_size = self._world_size_fn()
        multiple = self.make_vocab_size_divisible_by * world_size
        _dbg(
            "COMPUTE_MULTIPLE",
            divisible_by=self.make_vocab_size_divisible_by,
            world_size=world_size,
            multiple=multiple,
        )
        return multiple

    def compute(self, num_tokens: int) -> int:
        """纯函数：计算填充后词表大小，无副作用，不打印。

        [migrate ebbe40cd3] 对应上游 while 循环核心：
            after = num_tokens
            while (after % multiple) != 0:
                after += 1

        Walpurgis 改写点：
          - 改写为数学整除表达式（ceiling to multiple），去掉 while 循环；
            时间复杂度从 O(multiple) 降为 O(1)，语义不变。
          - 改写: after = num_tokens + (-num_tokens % multiple)
            等价于 math.ceil(num_tokens / multiple) * multiple，更直观。
          - 原词表已对齐时（dummy=0），行为与上游等价：不增加 token。
        """
        _dbg("COMPUTE_ENTRY", num_tokens=num_tokens)
        multiple = self._compute_multiple()
        # 改写：O(1) 整除对齐，替换上游 O(multiple) while 循环
        dummy_tokens = (-num_tokens) % multiple
        after = num_tokens + dummy_tokens
        _dbg(
            "COMPUTE_RESULT",
            num_tokens=num_tokens,
            multiple=multiple,
            dummy_tokens=dummy_tokens,
            after=after,
        )
        return after

    def apply(self, num_tokens: int) -> int:
        """计算填充后词表大小并打印摘要；完整对应上游 vocab_size_with_padding()。

        [migrate ebbe40cd3] 上游打印:
            '> padded vocab (size: {num_tokens}) with {dummy} dummy
             tokens (new size: {after})'
        Walpurgis 使用可注入的 rank_print_fn，默认行为与上游 print_rank_0 等价。
        """
        _dbg("APPLY_ENTRY", num_tokens=num_tokens)
        after = self.compute(num_tokens)
        dummy = after - num_tokens
        self._rank_print_fn(
            f"> padded vocab (size: {num_tokens}) with {dummy} dummy "
            f"tokens (new size: {after})"
        )
        _dbg("APPLY_DONE", before=num_tokens, dummy=dummy, after=after)
        return after


def vocab_size_with_padding(
    num_tokens: int,
    make_vocab_size_divisible_by: int,
    world_size_fn: Callable[[], int],
    rank_print_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """词表填充入口函数；语义等价于上游 megatron/utils.py vocab_size_with_padding()。

    [migrate ebbe40cd3] 上游签名: vocab_size_with_padding(num_tokens, args)
    Walpurgis 改写：将 args.make_vocab_size_divisible_by 和
    mpu.get_model_parallel_world_size 显式传入，消除对全局 args/mpu 的隐式依赖。
    调用方负责从 args 或分布式环境取值后传入，依赖关系在签名处可见。

    上游两处调用点（pretrain_bert.py / pretrain_gpt2.py）在 Walpurgis 中
    统一通过此函数入口调用，不再内联重复循环。

    Parameters
    ----------
    num_tokens : int
        原始词表大小（tokenizer.num_tokens）。
    make_vocab_size_divisible_by : int
        对应 args.make_vocab_size_divisible_by。
    world_size_fn : Callable[[], int]
        返回模型并行世界规模，对应 mpu.get_model_parallel_world_size。
    rank_print_fn : Callable[[str], None], optional
        打印函数，对应 print_rank_0；默认 print。

    Returns
    -------
    int
        填充后词表大小（≥ num_tokens，整除 divisible_by × world_size）。

    Examples
    --------
    >>> vocab_size_with_padding(30000, 128, lambda: 1)
    30080
    >>> vocab_size_with_padding(30080, 128, lambda: 1)
    30080
    """
    _dbg("ENTRY", num_tokens=num_tokens,
         make_vocab_size_divisible_by=make_vocab_size_divisible_by)
    policy = VocabPaddingPolicy(
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        world_size_fn=world_size_fn,
        rank_print_fn=rank_print_fn,
    )
    result = policy.apply(num_tokens)
    _dbg("EXIT", result=result)
    return result


__all__ = [
    "VocabPaddingPolicy",
    "vocab_size_with_padding",
]
