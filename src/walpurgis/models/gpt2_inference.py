"""
walpurgis/models/gpt2_inference.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit a1d04b793（第9个，共9062）
subject: Updating public repo with latest changes.

上游变更摘要（合并四个文件）
============================

generate_samples.py（+333 行，-84 行）——改动最大
  - 新增 top-k 采样（``--top-k``）、top-p 核采样（``--top-p``）
  - 新增 beam search（``--num-beams``）
  - 新增温度参数（``--temperature``）控制 logits 分布峭度
  - 新增批量生成接口（``generate_samples_batch()``）
  - 新增样本服务 HTTP server 框架（``generate_samples_server()``）
  - ``top_k_logits()`` 重构：支持 top-p（nucleus sampling），
    原来只支持 top-k

evaluate_gpt2.py（+43 行，-1 行）
  - 新增 ``--eval-seq-length`` 参数（支持评估时使用不同的序列长度）
  - 新增每步详细日志（step / loss / ppl）
  - 新增 ``--num-samples`` 评估样本数上限参数

model/gpt2_modeling.py（+42 行，-1 行）
  - ``GPT2ParallelSelfAttention`` 新增 ``attention_softmax_in_fp32`` 选项：
    在 fp16 训练时，attention score softmax 强制在 fp32 精度计算，
    避免数值溢出导致的 attention map 退化

mpu/transformer.py（+50 行，-17 行）
  - ``ParallelTransformerLayer`` 新增 ``apply_residual_connection_post_layernorm``：
    残差连接的加法在 LayerNorm 之后而非之前（Pre-LN vs Post-LN 结构可切换）
  - bias 初始化默认值调整（从 zero_init 改为正态分布初始化）

鲁迅拿法改写（≥20%）
=====================
generate_samples.py 是本次 commit 改动最大的文件（333 行净增量），
上游将 top-k、top-p、beam search、温度控制、批量生成、HTTP server
全部堆进一个单文件，如同《阿Q正传》里赵太爷家的院子——
每次来了新人（新功能），便再加一道门，门里套门，无人知道哪扇门通哪里。

鲁迅若见此文件，必会叹曰：「这不是院子，这是迷宫；
迷宫的建造者，早已忘记了自己当初为何要建迷宫。」

Walpurgis 将 generate_samples.py 的核心逻辑提炼为六层：

  1. ``SamplingStrategy`` 枚举 ——
     显式区分 GREEDY / TOP_K / TOP_P / BEAM_SEARCH 四种解码策略，
     使「用哪种方式生成」从运行时 if-else 链提升为可 match 的枚举，
     上游以多个 bool/int 参数的组合隐式决定策略。

  2. ``GenerationConfig`` dataclass ——
     封装全部生成超参：top_k / top_p / temperature / num_beams /
     max_new_tokens / out_seq_length，提供 ``strategy()`` 属性
     自动推断当前策略，``validate()`` 检查参数合理性（如 top_p ∈ (0, 1]）。

  3. ``LogitsFilter`` 类 ——
     对应上游重构后的 ``top_k_logits()``，将 top-k 过滤与 top-p
     核过滤合并为可组合的 pipeline，按 temperature → top_k → top_p
     顺序应用，每步均有 _dbg() 断点记录候选 token 数量变化。

  4. ``AttentionPrecisionConfig`` dataclass ——
     封装 model/gpt2_modeling.py 新增的 ``attention_softmax_in_fp32``
     选项，附带「为什么在 fp32」的数学原理注释（fp16 精度 softmax
     在大序列长度时易产生 attention sink，即极端的 0/1 分布）。

  5. ``ResidualConnectionMode`` 枚举 ——
     对应 mpu/transformer.py 新增的
     ``apply_residual_connection_post_layernorm``：
     PRE_LN（先 LN 再加残差，原始 GPT-2 结构）vs
     POST_LN（先加残差再 LN，原始 Transformer 结构）。

  6. ``EvalConfig`` dataclass ——
     封装 evaluate_gpt2.py 新增参数（eval_seq_length / num_samples）。

全链路 _dbg() 断点共 22 处，覆盖：
  MODULE_LOAD×2、SAMPLING_STRATEGY_ENUM_INIT、GEN_CFG_INIT、
  GEN_CFG_STRATEGY、GEN_CFG_VALIDATE_ERR、GEN_CFG_VALIDATE_OK、
  LOGITS_FILTER_INIT、LOGITS_FILTER_TEMP、LOGITS_FILTER_TOPK、
  LOGITS_FILTER_TOPK_APPLY、LOGITS_FILTER_TOPP、LOGITS_FILTER_TOPP_APPLY、
  LOGITS_FILTER_RESULT、ATTN_PREC_INIT、RESIDUAL_MODE_ENUM_INIT、
  EVAL_CFG_INIT、SELF_CHECK_START、SELF_CHECK_PASS×2、
  SELF_CHECK_LOGITS、SELF_CHECK_STRATEGY。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    if _DBG:
        print(f"[WALPURGIS-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "gpt2_inference.py 开始加载")

# ── 尝试导入 torch（允许在无 GPU 环境下加载模块）────────────────────────
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _dbg("MODULE_LOAD", "torch 不可用，LogitsFilter 将在无 tensor 模式运行")


# ─── 1. SamplingStrategy 枚举 ─────────────────────────────────────────────

class SamplingStrategy(Enum):
    """
    GPT-2 解码策略枚举。

    对应上游 generate_samples.py 通过参数组合隐式决定的解码路径：
      --top-k 0 --top-p 0.0 --num-beams 1  →  GREEDY
      --top-k K > 0                         →  TOP_K
      --top-p P ∈ (0, 1)                   →  TOP_P（核采样）
      --num-beams B > 1                     →  BEAM_SEARCH

    Walpurgis 将此组合逻辑显式化为枚举，使调用方通过
    ``GenerationConfig.strategy()`` 单一接口查询当前解码策略。
    """
    GREEDY = auto()       # 贪婪解码：argmax(logits)
    TOP_K = auto()        # Top-k 采样：从 top-k token 中按概率采样
    TOP_P = auto()        # 核采样（Nucleus Sampling）：动态候选集
    BEAM_SEARCH = auto()  # 集束搜索：维护 num_beams 条路径

    @property
    def description(self) -> str:
        descriptions = {
            SamplingStrategy.GREEDY: "贪婪解码 — argmax(logits)，确定性，速度最快",
            SamplingStrategy.TOP_K: "Top-k 采样 — 从得分最高的 k 个 token 中按概率采样",
            SamplingStrategy.TOP_P: "核采样 — 动态候选集，累积概率达到 p 截止",
            SamplingStrategy.BEAM_SEARCH: "集束搜索 — 维护多条候选路径，最终取最高得分",
        }
        return descriptions[self]


_dbg("SAMPLING_STRATEGY_ENUM_INIT",
     f"SamplingStrategy members: {[m.name for m in SamplingStrategy]}")


# ─── 2. GenerationConfig dataclass ───────────────────────────────────────

@dataclass
class GenerationConfig:
    """
    GPT-2 文本生成超参配置。

    字段来源（commit a1d04b793 generate_samples.py diff）
    ──────────────────────────────────────────────────────────
    top_k            --top-k        整数，Top-k 候选集大小（0 = 禁用）
    top_p            --top-p        浮点，核采样累积概率阈值（0.0 = 禁用）
    temperature      --temperature  浮点，logits 温度缩放（1.0 = 不变）
    num_beams        --num-beams    整数，beam search 路径数（1 = 贪婪/采样）
    max_new_tokens   --out-seq-length  最大新生成 token 数
    out_seq_length   原有参数，总输出序列长度（含提示）
    ──────────────────────────────────────────────────────────

    鲁迅观察：上游将 top_k=0 与 top_p=0.0 作为「禁用」信号，
    使参数的零值身兼两职——既是默认值，又是开关。
    Walpurgis 通过 ``strategy()`` 属性将此隐含逻辑显式化。
    """
    max_new_tokens: int = 128
    top_k: int = 0               # 0 表示禁用 top-k
    top_p: float = 0.0           # 0.0 表示禁用 top-p
    temperature: float = 1.0
    num_beams: int = 1           # 1 表示贪婪或单路径采样
    out_seq_length: int = 0      # 总输出长度（0 = 用 max_new_tokens）

    def __post_init__(self) -> None:
        _dbg("GEN_CFG_INIT",
             f"top_k={self.top_k} top_p={self.top_p} "
             f"temperature={self.temperature} num_beams={self.num_beams} "
             f"max_new_tokens={self.max_new_tokens}")
        self.validate()

    @property
    def strategy(self) -> SamplingStrategy:
        """
        推断当前解码策略（对应上游隐式参数组合逻辑）。

        优先级：BEAM_SEARCH > TOP_P > TOP_K > GREEDY
        （与上游 generate_samples.py 的 if-elif 顺序一致）
        """
        if self.num_beams > 1:
            s = SamplingStrategy.BEAM_SEARCH
        elif self.top_p > 0.0:
            s = SamplingStrategy.TOP_P
        elif self.top_k > 0:
            s = SamplingStrategy.TOP_K
        else:
            s = SamplingStrategy.GREEDY
        _dbg("GEN_CFG_STRATEGY",
             f"strategy={s.name}: {s.description}")
        return s

    def validate(self) -> None:
        """参数合理性检查。"""
        errors = []
        if self.top_p < 0.0 or self.top_p > 1.0:
            errors.append(f"top_p={self.top_p} 应在 [0.0, 1.0]")
        if self.temperature <= 0.0:
            errors.append(f"temperature={self.temperature} 必须 > 0")
        if self.top_k < 0:
            errors.append(f"top_k={self.top_k} 必须 ≥ 0")
        if self.num_beams < 1:
            errors.append(f"num_beams={self.num_beams} 必须 ≥ 1")
        if self.max_new_tokens <= 0:
            errors.append(f"max_new_tokens={self.max_new_tokens} 必须 > 0")
        if errors:
            _dbg("GEN_CFG_VALIDATE_ERR", "; ".join(errors))
            raise ValueError("GenerationConfig 校验失败:\n  " + "\n  ".join(errors))
        _dbg("GEN_CFG_VALIDATE_OK", "参数校验通过")


# ─── 3. LogitsFilter 类 ───────────────────────────────────────────────────

class LogitsFilter:
    """
    对应上游重构后的 ``top_k_logits()``（commit a1d04b793）。

    上游原函数仅支持 top-k，本次 commit 将其扩展为同时支持 top-p；
    但上游将两种过滤逻辑混写于一个函数中，条件分支复杂，难以单独测试。

    Walpurgis 将过滤拆分为三个串联步骤：
      Step 1: temperature scaling —— 调整 logits 峭度
      Step 2: top-k 过滤 —— 将 top-k 以外的 token 概率置为 -inf
      Step 3: top-p 核过滤 —— 在排序后的概率分布上按累积概率截断

    每步均有 _dbg() 断点，记录过滤前后候选 token 数量变化。

    注意：与 torch 强耦合，在无 torch 环境下仅提供配置接口。
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        _dbg("LOGITS_FILTER_INIT",
             f"strategy={config.strategy.name} "
             f"top_k={config.top_k} top_p={config.top_p} "
             f"temperature={config.temperature}")

    def apply(self, logits):
        """
        对 logits 张量（shape: [vocab_size] 或 [batch, vocab_size]）
        依次应用 temperature → top_k → top_p 过滤。

        Returns
        -------
        filtered logits（与输入同 shape）
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("LogitsFilter.apply() 需要 PyTorch")

        # Step 1: Temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
            _dbg("LOGITS_FILTER_TEMP",
                 f"temperature={self.config.temperature} applied")

        # Step 2: Top-k 过滤
        if self.config.top_k > 0:
            k = min(self.config.top_k, logits.size(-1))
            _dbg("LOGITS_FILTER_TOPK",
                 f"k={k} vocab_size={logits.size(-1)}")
            top_k_values = torch.topk(logits, k, dim=-1).values
            threshold = top_k_values[..., -1, None]
            filtered = logits < threshold
            logits = logits.masked_fill(filtered, float("-inf"))
            _dbg("LOGITS_FILTER_TOPK_APPLY",
                 f"masked {filtered.sum().item()} tokens to -inf")

        # Step 3: Top-p 核过滤（Nucleus Sampling）
        if self.config.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, dim=-1, descending=True
            )
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            _dbg("LOGITS_FILTER_TOPP",
                 f"top_p={self.config.top_p} computing cumulative probs")

            # 超过 top_p 的位置置为 -inf（保留刚好达到 top_p 的那个 token）
            sorted_indices_to_remove = cumulative_probs - F.softmax(
                sorted_logits, dim=-1
            ) > self.config.top_p
            sorted_logits = sorted_logits.masked_fill(
                sorted_indices_to_remove, float("-inf")
            )
            # 恢复原始 token 顺序
            logits = torch.zeros_like(logits).scatter_(
                -1, sorted_indices, sorted_logits
            )
            _dbg("LOGITS_FILTER_TOPP_APPLY",
                 f"masked {sorted_indices_to_remove.sum().item()} "
                 "tokens via nucleus filter")

        _dbg("LOGITS_FILTER_RESULT",
             f"final logits min={logits[logits != float('-inf')].min().item():.4f} "
             f"max={logits.max().item():.4f}"
             if _TORCH_AVAILABLE and logits.numel() > 0 else "")
        return logits

    def greedy_decode(self, logits) -> int:
        """贪婪解码：返回 argmax token id（标量 int）。"""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("需要 PyTorch")
        return int(torch.argmax(logits, dim=-1).item())

    def sample_decode(self, logits) -> int:
        """采样解码：经过 filter 后从 softmax 分布中采样。"""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("需要 PyTorch")
        filtered = self.apply(logits)
        probs = F.softmax(filtered, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


# ─── 4. AttentionPrecisionConfig dataclass ───────────────────────────────

@dataclass
class AttentionPrecisionConfig:
    """
    GPT-2 attention softmax 精度配置。

    对应上游 commit a1d04b793 在 model/gpt2_modeling.py 新增的
    ``attention_softmax_in_fp32`` 选项。

    数学背景
    ────────
    在 fp16 训练中，attention score 计算 ``QK^T / sqrt(d_k)`` 的数值范围
    可能超出 fp16 表示范围（特别是序列长度 > 2048、head_dim > 64 时）。
    强制在 fp32 精度执行 softmax 可避免：
      1. attention sink：单个 token 吸引几乎所有 attention（softmax 输出近似 one-hot）
      2. NaN/Inf 传播：fp16 overflow 导致梯度爆炸

    代价：fp32 softmax 消耗更多 HBM 带宽（~2× attention_scores 显存）。
    适用场景：长序列（> 2048）、大 head_dim（≥ 128）、或训练不稳定时。
    """
    softmax_in_fp32: bool = False   # 上游 attention_softmax_in_fp32

    def __post_init__(self) -> None:
        _dbg("ATTN_PREC_INIT",
             f"softmax_in_fp32={self.softmax_in_fp32} "
             + ("【启用】fp32 precision for attention softmax"
                if self.softmax_in_fp32 else
                "【禁用】attention softmax 使用模型默认精度"))

    def cast_for_softmax(self, scores):
        """
        在执行 softmax 前将 scores 转换到目标精度。

        Parameters
        ----------
        scores : torch.Tensor  attention scores，shape [..., seq_len, seq_len]

        Returns
        -------
        (cast_scores, original_dtype) 元组；
        cast_scores 已转换为目标精度，original_dtype 用于 softmax 后转回。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("需要 PyTorch")
        original_dtype = scores.dtype
        if self.softmax_in_fp32 and original_dtype != torch.float32:
            return scores.float(), original_dtype
        return scores, original_dtype

    def restore_after_softmax(self, probs, original_dtype):
        """softmax 后将 probs 转回原始精度（fp16 训练时恢复 fp16）。"""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("需要 PyTorch")
        if probs.dtype != original_dtype:
            return probs.to(original_dtype)
        return probs


# ─── 5. ResidualConnectionMode 枚举 ──────────────────────────────────────

class ResidualConnectionMode(Enum):
    """
    Transformer 残差连接位置（Pre-LN vs Post-LN）。

    对应上游 commit a1d04b793 在 mpu/transformer.py 新增的
    ``apply_residual_connection_post_layernorm`` 参数。

    PRE_LN（默认，apply_residual_connection_post_layernorm=False）：
        layernorm_output = LayerNorm(x)
        output = x + Attention(layernorm_output)          ← 残差加 x（LN 前）
        文献：GPT-2, GPT-3 的标准结构；训练更稳定，梯度流动更好。

    POST_LN（apply_residual_connection_post_layernorm=True）：
        layernorm_output = LayerNorm(x)
        output = LayerNorm(x) + Attention(layernorm_output)  ← 残差加 LN 输出
        文献：原始 Transformer（Vaswani 2017）的结构；
        在某些实验中有更好的收敛速度，但训练稳定性较差。

    Walpurgis 将 bool 参数替换为枚举，使含义自文档化。
    """
    PRE_LN = False    # apply_residual_connection_post_layernorm=False（上游默认）
    POST_LN = True    # apply_residual_connection_post_layernorm=True

    @classmethod
    def from_bool(cls, post_layernorm: bool) -> "ResidualConnectionMode":
        _dbg("RESIDUAL_MODE_ENUM_INIT",
             f"from_bool({post_layernorm}) → "
             f"{'POST_LN' if post_layernorm else 'PRE_LN'}")
        return cls.POST_LN if post_layernorm else cls.PRE_LN

    @property
    def description(self) -> str:
        if self == ResidualConnectionMode.PRE_LN:
            return "Pre-LN：残差连接在 LayerNorm 之前（GPT-2/3 标准结构，更稳定）"
        return "Post-LN：残差连接在 LayerNorm 之后（原始 Transformer 结构）"


# ─── 6. EvalConfig dataclass ─────────────────────────────────────────────

@dataclass
class EvalConfig:
    """
    GPT-2 评估配置（对应 evaluate_gpt2.py 新增参数）。

    字段来源（commit a1d04b793 evaluate_gpt2.py diff）
    ────────────────────────────────────────────────────────
    eval_seq_length   --eval-seq-length  评估时使用的序列长度
                                         （允许与训练时不同）
    num_samples       --num-samples      评估样本数上限（None = 全量）
    log_interval      原有参数，每 N 步打印一次评估进度
    ────────────────────────────────────────────────────────

    上游新增 eval_seq_length 的动机：
      评估集的文档长度分布可能与训练集不同；
      允许在评估时使用更长的序列可以更准确地测量 perplexity，
      而不受训练时序列长度截断的影响。
    """
    eval_seq_length: int = 1024
    num_samples: Optional[int] = None   # None = 评估全部样本
    log_interval: int = 100

    def __post_init__(self) -> None:
        if self.eval_seq_length <= 0:
            raise ValueError(
                f"eval_seq_length={self.eval_seq_length} 必须 > 0"
            )
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(
                f"num_samples={self.num_samples} 必须 > 0 或 None"
            )
        _dbg("EVAL_CFG_INIT",
             f"eval_seq_length={self.eval_seq_length} "
             f"num_samples={self.num_samples} "
             f"log_interval={self.log_interval}")

    def should_log(self, step: int) -> bool:
        """当前步是否应打印评估日志。"""
        return step % self.log_interval == 0

    def is_complete(self, step: int) -> bool:
        """是否已达到评估样本上限。"""
        if self.num_samples is None:
            return False
        return step >= self.num_samples


# ─── 自检 ─────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    9 项断言，覆盖 SamplingStrategy 推断、GenerationConfig 校验、
    LogitsFilter（无 torch 时跳过）、AttentionPrecisionConfig、
    ResidualConnectionMode、EvalConfig。
    """
    _dbg("SELF_CHECK_START", "开始 self_check()")

    # 1. 贪婪策略推断
    cfg = GenerationConfig(top_k=0, top_p=0.0, num_beams=1)
    assert cfg.strategy == SamplingStrategy.GREEDY, f"期望 GREEDY，得 {cfg.strategy}"

    # 2. Top-k 策略推断
    cfg = GenerationConfig(top_k=50, top_p=0.0, num_beams=1)
    assert cfg.strategy == SamplingStrategy.TOP_K
    _dbg("SELF_CHECK_STRATEGY", "TOP_K OK")

    # 3. Top-p 策略推断（top_p 优先于 top_k 0，但低于 num_beams）
    cfg = GenerationConfig(top_k=0, top_p=0.9, num_beams=1)
    assert cfg.strategy == SamplingStrategy.TOP_P

    # 4. Beam search 策略推断（最高优先级）
    cfg = GenerationConfig(top_k=50, top_p=0.9, num_beams=4)
    assert cfg.strategy == SamplingStrategy.BEAM_SEARCH

    # 5. GenerationConfig 参数校验（top_p 越界）
    try:
        GenerationConfig(top_p=1.5)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 6. LogitsFilter（仅 torch 可用时）
    if _TORCH_AVAILABLE:
        import torch
        cfg = GenerationConfig(top_k=5, temperature=0.8)
        f = LogitsFilter(cfg)
        logits = torch.randn(100)
        filtered = f.apply(logits)
        # 过滤后，-inf 以外的有限值数量应 ≤ top_k
        finite = (filtered != float("-inf")).sum().item()
        assert finite <= cfg.top_k, f"top_k 过滤后有限值数={finite} > k={cfg.top_k}"
        _dbg("SELF_CHECK_LOGITS", f"LogitsFilter OK, finite={finite}")

    # 7. AttentionPrecisionConfig
    apc = AttentionPrecisionConfig(softmax_in_fp32=False)
    assert apc.softmax_in_fp32 is False

    # 8. ResidualConnectionMode.from_bool
    assert ResidualConnectionMode.from_bool(False) == ResidualConnectionMode.PRE_LN
    assert ResidualConnectionMode.from_bool(True) == ResidualConnectionMode.POST_LN

    # 9. EvalConfig.is_complete
    ec = EvalConfig(num_samples=100)
    assert not ec.is_complete(99)
    assert ec.is_complete(100)

    _dbg("SELF_CHECK_PASS", "全部 9 项断言通过")
    print("[gpt2_inference.self_check] OK — 9 assertions passed", file=sys.stderr)
    return True


_dbg("MODULE_LOAD", "gpt2_inference.py 加载完成")

if __name__ == "__main__":
    self_check()
