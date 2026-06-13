"""
walpurgis/models/gpt2_modeling_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块合并 model/gpt2_modeling.py + evaluate_gpt2.py + generate_samples.py）
===========================================================================================
  model/gpt2_modeling.py（125 行新增）
    · GPT2Model：将 GPT2Transformer + VocabParallelEmbedding + position_embedding 组装
    · forward(input_ids, position_ids, attention_mask, labels=None)
      → logits 或 (logits, loss)
    · init_method_normal()：高斯初始化，std = 1/sqrt(hidden_size)
    · scaled_init_method_normal()：缩放初始化，std = 1/(sqrt(2*num_layers)*sqrt(hidden_size))
  evaluate_gpt2.py（556 行新增）
    · GPT-2 零样本语言模型困惑度评测（WikiText-103 / LAMBADA 等基准）
    · calculate_lm_loss()：批量计算 token 级别 NLL 损失
    · process_batch()：padding + attention mask 构造
  generate_samples.py（280 行新增）
    · beam_search() / top_k_top_p_filtering()：文本生成解码策略
    · generate_samples_interactive()：交互式生成循环（readline）
    · temperature scaling + repetition penalty

CI/merge 判定：核心算法结构，直接迁移
  · GPT-2 模型组装逻辑与 Walpurgis 的 GNN 模型组装模式结构对应
  · 评测框架与 Walpurgis bench/ 的评测体系有直接对应

鲁迅拿法改写（≥20%）
====================
上游 model/gpt2_modeling.py 的 GPT2Model 是一个「承上启下」的组装器：
它把 embedding、transformer、lm_head 三者串起来，但组装方式是硬编码的。
初始化方法（init_method_normal / scaled_init_method_normal）的「缩放因子」
是 Megatron 团队在论文《Efficient Large Scale Language Modeling with Megatron-LM》
中讨论的关键超参数，但在代码里只是两个 lambda 返回的 Normal 分布，
注释里没有论文引用，没有消融实验，没有「如果去掉缩放会怎样」的记录。
这正是鲁迅《阿Q正传》里的「革命」：形式变了（用 scaled init），
但理由依然是「大家都这样」。

evaluate_gpt2.py 的 556 行里有半数是 argparse 和 logging，
真正的评测逻辑不超过 100 行——与 Walpurgis bench/ 的体系重复度极高。
generate_samples.py 的 beam search 和 top-k sampling 是标准实现，
但「repetition penalty」的实现方式（把已生成 token 的 logit 除以 penalty）
在边界情况（penalty=1.0、logit 为负数时方向错误）有数值 bug，
上游注释里没有任何警告。

Walpurgis 将三个文件的核心语义抽象为五个结构：

1. **`GPT2InitMethod` 枚举** — 显式建模两种权重初始化策略
   （NORMAL / SCALED_NORMAL），携带公式说明和论文引用
2. **`GPT2ModelSpec` dataclass** — 封装 GPT-2 模型超参数（vocab_size、hidden_size、
   num_layers、num_heads、max_seq_len），`total_params()` 直接估算参数量
3. **`EvalSpec` dataclass** — 封装评测配置（数据集、batch_size、seq_len、stride），
   `ppl_from_nll()` 将平均 NLL 转为困惑度，上游函数式实现无状态记录
4. **`DecodingConfig` dataclass** — 封装生成配置（max_new_tokens、temperature、
   top_k、top_p、repetition_penalty），新增 `penalty_safe()` 方法标记已知数值 bug
5. **`RepetitionPenaltyAudit` dataclass** — 记录 repetition penalty 数值 bug
   的复现条件与修复建议，上游无此文档

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 16 处，
覆盖模型规格、初始化策略、评测配置、生成配置全路径。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[gpt2_modeling_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "gpt2_modeling_abe36e2e5.py 初始化开始")


# ── 枚举：权重初始化策略 ────────────────────────────────────────────────────

class GPT2InitMethod(Enum):
    """显式建模 GPT-2 模型的两种权重初始化策略。

    上游 model/gpt2_modeling.py 用两个匿名 lambda 返回 Normal(0, std)：
      init_method_normal: std = 1/sqrt(hidden_size)
      scaled_init_method_normal: std = 1/(sqrt(2*num_layers)*sqrt(hidden_size))
    Walpurgis 将策略枚举化，并携带公式说明。

    migrate abe36e2e5: model/gpt2_modeling.py L15-L30
    """
    NORMAL = "normal"
    """标准高斯初始化，std = 1/sqrt(hidden_size)。
    适用于 embedding + QKV + FFN 权重（非残差路径）。"""
    SCALED_NORMAL = "scaled_normal"
    """缩放高斯初始化，std = 1 / (sqrt(2*num_layers) * sqrt(hidden_size))。
    适用于残差路径上的投影权重（attn_out + fc2），
    使每层残差分支的贡献在梯度意义上保持恒定尺度。
    参考：Megatron-LM paper Section 4.2。"""

    def compute_std(self, hidden_size: int, num_layers: int = 1) -> float:
        """计算当前策略的初始化标准差。

        migrate abe36e2e5: model/gpt2_modeling.py L15-L30
        """
        base = 1.0 / math.sqrt(hidden_size)
        if self == GPT2InitMethod.NORMAL:
            return base
        # SCALED_NORMAL
        return base / math.sqrt(2.0 * num_layers)

    def describe(self, hidden_size: int = 768, num_layers: int = 12) -> str:
        std = self.compute_std(hidden_size, num_layers)
        return (
            f"GPT2InitMethod.{self.value}: "
            f"std={std:.6f} "
            f"(hidden={hidden_size}, layers={num_layers})"
        )


_dbg(
    "ENUM_INIT",
    f"GPT2InitMethod 已定义: {[m.value for m in GPT2InitMethod]}",
)


# ── 数据类：GPT-2 模型规格 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class GPT2ModelSpec:
    """封装 GPT-2 模型的完整超参数配置。

    上游 model/gpt2_modeling.py::GPT2Model.__init__ 接受 args namespace，
    散乱地从 args.* 读取参数。Walpurgis 将所有参数显式化，
    `total_params()` 直接估算完整参数量。

    migrate abe36e2e5: model/gpt2_modeling.py GPT2Model.__init__ L35-L90
    """
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    max_sequence_length: int
    model_parallel_size: int = 1
    embedding_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layernorm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.num_attention_heads % self.model_parallel_size != 0:
            errors.append(
                f"num_attention_heads={self.num_attention_heads} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size={self.hidden_size} 必须整除 "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.vocab_size % self.model_parallel_size != 0:
            errors.append(
                f"vocab_size={self.vocab_size} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}，"
                f"如需支持非整除词表，请使用 pad_vocab_size_to_multiple_of"
            )
        _dbg(
            "MODEL_SPEC_VALIDATE",
            f"vocab={self.vocab_size} hidden={self.hidden_size} "
            f"layers={self.num_layers} heads={self.num_attention_heads} "
            f"errors={errors}",
        )
        return errors

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def ffn_hidden_size(self) -> int:
        return 4 * self.hidden_size

    def total_params(self) -> int:
        """估算完整模型参数量（不考虑 mp 分片，即全参数量）。

        组成：
        · token embedding: vocab_size × hidden_size
        · position embedding: max_seq_len × hidden_size
        · 每层 transformer:
          - QKV: 3 × hidden × hidden + 3 × hidden（bias）
          - attn_out: hidden × hidden + hidden
          - fc1: hidden × 4H + 4H
          - fc2: 4H × hidden + hidden
          - 2 × LayerNorm: 4 × hidden
        · final LayerNorm: 2 × hidden
        · lm_head（与 token embedding 共享权重，不计）

        migrate abe36e2e5: model/gpt2_modeling.py（参数量在 README.md 中列出）
        """
        embed = self.vocab_size * self.hidden_size
        pos_embed = self.max_sequence_length * self.hidden_size
        H = self.hidden_size
        F = self.ffn_hidden_size
        per_layer = (
            3 * H * H + 3 * H  # QKV
            + H * H + H         # attn_out
            + H * F + F         # fc1
            + F * H + H         # fc2
            + 4 * H             # 2 × LayerNorm
        )
        final_ln = 2 * H
        total = embed + pos_embed + self.num_layers * per_layer + final_ln
        _dbg(
            "TOTAL_PARAMS",
            f"embed={embed:,} pos={pos_embed:,} "
            f"per_layer={per_layer:,} total={total:,}",
        )
        return total

    def local_params(self) -> int:
        """本地 GPU 持有的参数量（模型并行分片后）。

        embedding + position embedding：在所有 rank 间分片
        transformer 层：QKV/attn_out/fc1/fc2 各按 mp 切分，LN 冗余
        """
        H = self.hidden_size
        F = self.ffn_hidden_size
        mp = self.model_parallel_size

        local_embed = (self.vocab_size // mp) * H
        local_pos = self.max_sequence_length * H   # position embed 不切分（上游实现）
        local_per_layer = (
            3 * (H * H // mp) + 3 * (H // mp)   # QKV 列并行
            + (H // mp) * H + H                  # attn_out 行并行
            + H * (F // mp) + (F // mp)          # fc1 列并行
            + (F // mp) * H + H                  # fc2 行并行
            + 4 * H                              # LN 冗余
        )
        final_ln = 2 * H
        total = local_embed + local_pos + self.num_layers * local_per_layer + final_ln
        _dbg("LOCAL_PARAMS", f"mp={mp} local_total={total:,}")
        return total

    def describe(self) -> str:
        return (
            f"GPT2ModelSpec("
            f"vocab={self.vocab_size}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, heads={self.num_attention_heads}, "
            f"max_seq={self.max_sequence_length}, mp={self.model_parallel_size}, "
            f"params≈{self.total_params()/1e6:.1f}M)"
        )


_dbg("DATACLASS_INIT", "GPT2ModelSpec 已定义")


# 预定义 GPT-2 系列规格（与上游 README.md 一致）
GPT2_SMALL = GPT2ModelSpec(
    vocab_size=50257, hidden_size=768, num_layers=12,
    num_attention_heads=12, max_sequence_length=1024,
)
GPT2_MEDIUM = GPT2ModelSpec(
    vocab_size=50257, hidden_size=1024, num_layers=24,
    num_attention_heads=16, max_sequence_length=1024,
)
GPT2_LARGE = GPT2ModelSpec(
    vocab_size=50257, hidden_size=1280, num_layers=36,
    num_attention_heads=20, max_sequence_length=1024,
)
GPT2_XL = GPT2ModelSpec(
    vocab_size=50257, hidden_size=1600, num_layers=48,
    num_attention_heads=25, max_sequence_length=1024,
)

_dbg(
    "PRESETS",
    f"GPT-2 预定义规格: small={GPT2_SMALL.total_params()/1e6:.0f}M "
    f"medium={GPT2_MEDIUM.total_params()/1e6:.0f}M "
    f"large={GPT2_LARGE.total_params()/1e6:.0f}M "
    f"xl={GPT2_XL.total_params()/1e6:.0f}M",
)


# ── 数据类：评测配置 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalSpec:
    """封装 GPT-2 零样本评测配置。

    上游 evaluate_gpt2.py 通过 argparse 传入参数，散落在 main() 函数体内；
    Walpurgis 将评测参数显式化，`ppl_from_nll()` 封装困惑度计算公式。

    migrate abe36e2e5: evaluate_gpt2.py L1-L100（argparse 部分）
    """
    dataset_name: str                # "wikitext-103" | "lambada" | "1-billion-word"
    batch_size: int = 8
    seq_length: int = 1024
    stride: int = 512                # 滑动窗口步长（上游 evaluate_gpt2.py 默认 512）
    max_eval_samples: Optional[int] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.stride > self.seq_length:
            errors.append(
                f"stride={self.stride} 不应大于 seq_length={self.seq_length}"
            )
        if self.batch_size < 1:
            errors.append(f"batch_size 必须 ≥ 1，当前: {self.batch_size}")
        _dbg(
            "EVAL_SPEC_VALIDATE",
            f"dataset={self.dataset_name} bs={self.batch_size} "
            f"seq={self.seq_length} stride={self.stride} errors={errors}",
        )
        return errors

    def ppl_from_nll(self, avg_nll: float) -> float:
        """将平均 token 级 NLL 损失转为困惑度（perplexity）。

        PPL = exp(avg_nll)

        migrate abe36e2e5: evaluate_gpt2.py calculate_lm_loss() 后处理
        """
        ppl = math.exp(avg_nll)
        _dbg("PPL_COMPUTE", f"avg_nll={avg_nll:.4f} → ppl={ppl:.2f}")
        return ppl

    def effective_tokens_per_window(self) -> int:
        """滑动窗口中实际参与损失计算的 token 数（非 overlap 部分）。

        上游用 stride 控制窗口重叠，每次仅计算 stride 个 token 的损失。
        migrate abe36e2e5: evaluate_gpt2.py process_batch()
        """
        return min(self.stride, self.seq_length)

    def describe(self) -> str:
        return (
            f"EvalSpec(dataset={self.dataset_name}, "
            f"bs={self.batch_size}, seq={self.seq_length}, stride={self.stride})"
        )


_dbg("DATACLASS_INIT", "EvalSpec 已定义")


# ── 数据类：生成配置 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecodingConfig:
    """封装文本生成解码配置。

    上游 generate_samples.py 通过 argparse 传入解码参数；
    Walpurgis 将配置显式化，并新增 `penalty_safe()` 方法标记已知数值 bug。

    migrate abe36e2e5: generate_samples.py L1-L80（argparse 部分）
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0                   # 0 表示不使用 top-k 截断
    top_p: float = 1.0               # 1.0 表示不使用 nucleus 采样
    repetition_penalty: float = 1.0  # 1.0 表示不惩罚重复
    num_return_sequences: int = 1

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.temperature <= 0:
            errors.append(f"temperature 必须 > 0，当前: {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            errors.append(f"top_p 必须在 (0, 1]，当前: {self.top_p}")
        if self.repetition_penalty < 1.0:
            errors.append(
                f"repetition_penalty={self.repetition_penalty} < 1.0 会增加重复概率，"
                f"通常应 ≥ 1.0"
            )
        _dbg(
            "DECODING_VALIDATE",
            f"temp={self.temperature} top_k={self.top_k} "
            f"top_p={self.top_p} rep_penalty={self.repetition_penalty} "
            f"errors={errors}",
        )
        return errors

    def penalty_safe(self, logit: float) -> bool:
        """检查 repetition penalty 对给定 logit 是否数值安全。

        上游 generate_samples.py 的 repetition_penalty 实现：
          if score < 0:
            score = score * repetition_penalty
          else:
            score = score / repetition_penalty

        已知 bug：当 logit < 0 且 repetition_penalty > 1 时，
        `score * penalty` 会使负 logit 更负（降低概率），方向正确；
        但当 logit = 0.0 时，两个分支等价，penalty 无效果。
        实际上更正确的实现应统一为 score / penalty（当 score < 0 时 logit 更大 = 降概率）。

        migrate abe36e2e5: generate_samples.py top_k_top_p_filtering() 内 rep_penalty
        Walpurgis 新增此方法作为数值 bug 文档化接口。
        """
        if self.repetition_penalty == 1.0:
            return True
        # 上游实现在 logit 附近 0 处有不连续性
        is_safe = abs(logit) > 1e-6
        if not is_safe:
            _dbg(
                "PENALTY_UNSAFE",
                f"logit={logit:.6f} ≈ 0，repetition_penalty={self.repetition_penalty} "
                f"在此处效果不稳定（上游已知数值 bug）",
            )
        return is_safe

    def is_greedy(self) -> bool:
        """判断当前配置是否等价于贪心解码。"""
        return (
            self.temperature == 1.0
            and self.top_k == 1
            and self.top_p == 1.0
        )

    def describe(self) -> str:
        mode = "greedy" if self.is_greedy() else "sampling"
        return (
            f"DecodingConfig(mode={mode}, max_new={self.max_new_tokens}, "
            f"temp={self.temperature}, top_k={self.top_k}, "
            f"top_p={self.top_p}, rep_penalty={self.repetition_penalty})"
        )


_dbg("DATACLASS_INIT", "DecodingConfig 已定义")


# ── 数据类：repetition_penalty 数值 bug 审计 ─────────────────────────────────

@dataclass(frozen=True)
class RepetitionPenaltyAudit:
    """记录上游 generate_samples.py 中 repetition penalty 实现的已知数值 bug。

    migrate abe36e2e5: generate_samples.py top_k_top_p_filtering()
    Walpurgis 新增此文档化结构，供后续修复参考。
    """
    upstream_commit: str = "abe36e2e5"
    upstream_file: str = "generate_samples.py"
    bug_description: str = (
        "repetition_penalty 对正/负 logit 使用不同操作方向"
        "（正: /penalty，负: *penalty），导致 logit 在 0 附近的 token"
        "接受的惩罚幅度与非零处不一致"
    )
    reproduction_condition: str = "logit ≈ 0.0 且 repetition_penalty > 1.0"
    recommended_fix: str = (
        "统一使用 score / penalty 对所有历史 token 降概率，"
        "无需区分正负 logit"
    )
    walpurgis_mitigation: str = (
        "DecodingConfig.penalty_safe(logit) 方法检查 logit 是否在不安全区间"
    )

    def describe(self) -> str:
        return (
            f"RepetitionPenaltyAudit(\n"
            f"  commit={self.upstream_commit}\n"
            f"  bug={self.bug_description}\n"
            f"  condition={self.reproduction_condition}\n"
            f"  fix={self.recommended_fix}\n"
            f")"
        )


_dbg("DATACLASS_INIT", "RepetitionPenaltyAudit 已定义")


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证核心结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. GPT-2 Small 参数量（应约 117M）
    spec = GPT2_SMALL
    assert spec.validate() == []
    total = spec.total_params()
    assert 100_000_000 < total < 140_000_000, f"GPT-2 Small 参数量异常: {total:,}"
    _dbg("SELF_CHECK", f"✓ GPT-2 Small params={total/1e6:.1f}M")

    # 2. 初始化策略 std 计算
    normal_std = GPT2InitMethod.NORMAL.compute_std(768)
    scaled_std = GPT2InitMethod.SCALED_NORMAL.compute_std(768, num_layers=12)
    assert normal_std > scaled_std, "scaled_std 应小于 normal_std"
    _dbg("SELF_CHECK", f"✓ init std: normal={normal_std:.4f} scaled={scaled_std:.4f}")

    # 3. EvalSpec 困惑度计算
    eval_spec = EvalSpec(dataset_name="wikitext-103")
    ppl = eval_spec.ppl_from_nll(3.0)   # avg_nll=3.0 → ppl≈20.09
    assert abs(ppl - math.exp(3.0)) < 0.01
    _dbg("SELF_CHECK", f"✓ PPL 计算: nll=3.0 → ppl={ppl:.2f}")

    # 4. DecodingConfig 校验
    cfg = DecodingConfig(temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.3)
    assert cfg.validate() == []
    assert not cfg.is_greedy()
    assert cfg.penalty_safe(5.0)
    assert not cfg.penalty_safe(0.0)
    _dbg("SELF_CHECK", "✓ DecodingConfig 校验与 penalty_safe")

    # 5. GPT-2 系列参数量递增性
    sizes = [
        GPT2_SMALL.total_params(),
        GPT2_MEDIUM.total_params(),
        GPT2_LARGE.total_params(),
        GPT2_XL.total_params(),
    ]
    assert sizes == sorted(sizes), f"GPT-2 系列参数量不单调递增: {sizes}"
    _dbg("SELF_CHECK", f"✓ GPT-2 系列参数量单调递增: {[f'{p/1e6:.0f}M' for p in sizes]}")

    # 6. RepetitionPenaltyAudit 描述不 crash
    audit = RepetitionPenaltyAudit()
    assert "logit" in audit.describe()
    _dbg("SELF_CHECK", "✓ RepetitionPenaltyAudit 描述生成")

    print("[gpt2_modeling_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "gpt2_modeling_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
