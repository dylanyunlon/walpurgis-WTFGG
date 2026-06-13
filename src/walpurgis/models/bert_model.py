"""
walpurgis/models/bert_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 73af12903 (第24个, 共9062)
Subject: "Major refactoring, combining gpt2 and bert"

上游改动摘要
============
megatron/model/bert_model.py (218行新增):
  73af12903 将 pretrain_bert.py 中散落的 BertModel 构造逻辑
  集中到此文件，成为共享 language_model 基础设施之上的薄包装层。

  上游 BertModel 的组成:
    - get_language_model(add_pooler=True, num_tokentypes=2, ...)
      → LanguageModel (embedding + transformer + pooler)
    - bert_extended_attention_mask(): 将 [PAD] 位置的 mask 值
      从 0/1 转换为 -10000.0 / 0.0 (attention score 的加法 mask)
    - bert_position_ids(): 生成 [0, 1, 2, ..., seq_len-1] 位置 ID
    - BertLMHead: word embedding weight 的 tied 输出头 (MLM)
    - BertModel: 将上述组件组装为完整 BERT 前向计算图

  上游 megatron/model/model.py (90行，已删除):
    - 原 BertModel/GPT2Model 的「模型组装器」角色被此文件取代。
    - 原文件导出的 gpt2_get_params_for_weight_decay_optimization()
      移入 megatron/model/utils.py。

  上游 megatron/model/__init__.py 同步更新:
    - 删除 from megatron.model.modeling import BertModel
    - 新增 from megatron.model.bert_model import BertModel
    - 删除 gpt2_get_params_for_weight_decay_optimization 导出
      (移入 utils.py，evaluate_gpt2.py 对应删除该 import)

kicker: bert_extended_attention_mask 的 mask 转换:
  上游: (attention_mask.unsqueeze(1).unsqueeze(2)
        .to(dtype=next(self.parameters()).dtype))
  意图: 将 1/0 的 mask 转为注意力分数的加法项
  实现: 1.0 → 0.0 (可见), 0.0 → -10000.0 (被遮蔽)
  细节: 使用 (1 - mask) * -10000.0 的计算形式

鲁迅拿法改写（≥20%）
=====================
鲁迅说:「只要一本书上写着，人们便以为是好的。」

上游 BertModel 是一个「组装器」——
它把 language_model 的各个零件拼在一起，
但它本身并不解释这些零件为何以这种方式拼合。
bert_extended_attention_mask 将 0/1 mask 转换为 -10000/0 加法项，
但没有注释说明为什么是 -10000 而不是 -inf？
为什么要用 dtype=next(self.parameters()).dtype 而不是直接 float？
BertLMHead 为什么要 tie word embedding weight？
这些设计决策都是「默认如此」，如同《孔乙己》里的茴香豆:
有四种写法，但没有人告诉你为什么需要四种写法。

Walpurgis 将 BertModel 的「设计决策」改写为三个显式组件:

1. **`BertMaskConvention` dataclass** — 显式建模 BERT attention mask
   的数值转换约定: 从 {0, 1} 到加法 mask 的变换规则，
   以及 -10000.0 这个「大负数」的来历（近似 -inf，但不会造成
   fp16 下的 NaN）。

2. **`BertHeadPolicy` 枚举** — 显式区分 BERT 的两种输出头:
   MLM_HEAD (MaskedLM，BertLMHead，weight-tied) 和
   NSP_HEAD (NextSentencePrediction，Pooler，用 [CLS] token)。
   上游在同一 BertModel 中隐式包含两者，无枚举区分。

3. **`BertModelManifest`** — 建模 73af12903 引入的 BertModel
   组件清单，记录上游文件变更、Walpurgis 迁移位置，
   以及 BERT-specific 设计决策的审计接口。

全链路 _dbg() 断点共 **12 处**:
MODULE_LOAD, MASK_CONV_INIT, MASK_CONV_VALIDATE, MASK_CONV_APPLY,
HEAD_POLICY_ENUM, MANIFEST_INIT, MANIFEST_AUDIT_START,
MANIFEST_AUDIT_MASK, MANIFEST_AUDIT_HEAD, MANIFEST_AUDIT_DONE,
SELF_CHECK_START, SELF_CHECK_PASS。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0").strip()
_DEBUG = _DEBUG_ENV in ("1", "bert_model")


def _dbg(tag: str, msg: object = "") -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出结构化诊断行到 stderr"""
    if _DEBUG:
        print(f"[BERT-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "bert_model 加载 — 73af12903 BERT 薄包装层建模")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. BertMaskConvention — attention mask 数值转换约定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class BertMaskConvention:
    """
    BERT attention mask 的数值转换约定。

    上游 bert_extended_attention_mask() 的核心逻辑:
      attention_mask_adder = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2))
      attention_mask_adder = attention_mask_adder * -10000.0

    此转换将:
      输入 mask 值 1 (可见位置) → adder = 0.0  (注意力分数不变)
      输入 mask 值 0 ([PAD] 位置) → adder = -10000.0 (注意力分数极小)

    为什么是 -10000.0 而不是 -inf？
      fp16 中 -inf 会导致 softmax 输出 NaN；
      -10000.0 在 softmax 后足够接近 0，且不产生 NaN。

    鲁迅视角：-10000.0 是一个「善意的谎言」——
    它不说「我无法参与注意力计算」，而是说「我的注意力分数极低」。
    效果等价，但避免了 fp16 数值崩溃的陷阱。
    """
    visible_input_value: float = 1.0    # 输入 mask 中「可见」位置的值
    masked_input_value:  float = 0.0    # 输入 mask 中「被遮蔽」位置的值
    additive_visible:    float = 0.0    # 转换后可见位置加到 attention score 的值
    additive_masked:     float = -10000.0  # 转换后遮蔽位置加到 attention score 的值
    # 为什么不用 -inf
    avoid_inf_reason: str = (
        "fp16 下 -inf + score = -inf，softmax(-inf) = NaN；"
        "-10000.0 足够小使 softmax 输出趋近 0，且不产生 NaN。"
        "[fix: 73af12903 继承自 modeling.py 的历史约定]"
    )

    def __post_init__(self) -> None:
        _dbg("MASK_CONV_INIT", (
            f"visible={self.visible_input_value}→{self.additive_visible}, "
            f"masked={self.masked_input_value}→{self.additive_masked}"
        ))
        self._validate()

    def _validate(self) -> None:
        """验证约定的数学正确性"""
        _dbg("MASK_CONV_VALIDATE", "验证 mask 转换约定")
        assert self.additive_visible == 0.0, (
            "可见位置的 additive mask 必须为 0.0 (不影响 attention score)"
        )
        assert self.additive_masked < -1000.0, (
            f"遮蔽位置的 additive mask 必须足够小 (<-1000)，"
            f"得到 {self.additive_masked}"
        )
        assert self.additive_masked > float('-inf'), (
            "遮蔽位置的 additive mask 不应为 -inf (fp16 NaN 风险)"
        )
        _dbg("MASK_CONV_VALIDATE", "约定验证通过 ✓")

    def transform_description(self) -> str:
        """返回人类可读的转换描述"""
        _dbg("MASK_CONV_APPLY", "生成 mask 转换描述")
        return (
            f"bert_extended_attention_mask 转换:\n"
            f"  {self.visible_input_value} (可见) → {self.additive_visible} (加到 attention score)\n"
            f"  {self.masked_input_value} ([PAD]) → {self.additive_masked} (使注意力权重趋近 0)\n"
            f"  实现: (1.0 - mask.unsqueeze(1).unsqueeze(2)) * {self.additive_masked}\n"
            f"  避免 -inf 原因: {self.avoid_inf_reason}"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "visible_input_value":  self.visible_input_value,
            "masked_input_value":   self.masked_input_value,
            "additive_visible":     self.additive_visible,
            "additive_masked":      self.additive_masked,
            "avoid_inf_reason":     self.avoid_inf_reason,
            "transform_formula":    "(1.0 - mask) * additive_masked",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BertHeadPolicy — BERT 输出头策略枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BertHeadPolicy(Enum):
    """
    BERT 输出头的功能分类。

    上游 BertModel 同时包含两个输出头:
      1. BertLMHead: 用于 Masked LM (MLM) 预训练任务
         - 输入: transformer 全部 hidden_states
         - 输出: [batch, seq_len, vocab_size] logits
         - 特性: weight 与 word embedding 共享 (weight tying)
      2. Pooler (隐含在 LanguageModel 中): 用于 NSP 任务
         - 输入: [CLS] token 的 hidden_state
         - 输出: [batch, hidden_size]，接 2-class 分类头

    上游将两者在 BertModel.forward() 中同时计算，
    无法按任务选择性关闭其中一个。

    鲁迅视角：BERT 的两个头像两个徒弟——
    一个负责「完形填空」(MLM)，一个负责「判断上下文」(NSP)。
    上游让他们永远绑在一起出工，
    不管你用不用 NSP，Pooler 都在那里占着显存。
    枚举给他们发了独立的证书，以备将来分工而治。
    """
    MLM_HEAD = auto()  # Masked LM head: weight-tied to word embedding
    NSP_HEAD  = auto()  # Next Sentence Prediction head: [CLS] + pooler

    @property
    def uses_weight_tying(self) -> bool:
        """True → 此头部的输出权重与 word embedding 共享"""
        return self is BertHeadPolicy.MLM_HEAD

    @property
    def input_token(self) -> str:
        """描述此头使用的 token 输入"""
        return {
            BertHeadPolicy.MLM_HEAD: "全部 token 的 hidden_states",
            BertHeadPolicy.NSP_HEAD: "[CLS] token (index 0) 的 hidden_state",
        }[self]

    @property
    def output_description(self) -> str:
        return {
            BertHeadPolicy.MLM_HEAD: "[batch, seq_len, vocab_size] logits",
            BertHeadPolicy.NSP_HEAD: "[batch, 2] 二分类 logits",
        }[self]


_dbg("HEAD_POLICY_ENUM",
     f"BertHeadPolicy: {[p.name for p in BertHeadPolicy]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. BertModelManifest — 73af12903 BertModel 组件清单
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BertModelManifest:
    """
    73af12903 引入的 megatron/model/bert_model.py 组件清单。

    记录上游文件结构、BERT-specific 设计决策、
    以及 Walpurgis 迁移位置。
    """
    upstream_commit: str = "73af12903"
    upstream_file: str = "megatron/model/bert_model.py"
    upstream_lines: int = 218
    walpurgis_file: str = "src/walpurgis/models/bert_model.py"

    # 上游文件中的主要组件
    UPSTREAM_COMPONENTS: Tuple[str, ...] = field(default_factory=lambda: (
        "bert_extended_attention_mask(attention_mask) → extended_mask",
        "bert_position_ids(token_ids) → position_ids",
        "BertLMHead.__init__(mpu, hidden_size, vocab_size, init_method)",
        "BertLMHead.forward(hidden_states, word_embeddings_weight) → logits",
        "BertModel.__init__(num_tokentypes, parallel_output)",
        "BertModel.forward(input_ids, attention_mask, tokentype_ids, layer_past) → lm_logits, pooled_output",
    ))

    # 73af12903 同步修改的相关文件
    RELATED_CHANGES: Tuple[Tuple[str, str], ...] = field(default_factory=lambda: (
        ("megatron/model/__init__.py",
         "BertModel import 路径: modeling → bert_model; 删除 gpt2_get_params_for_weight_decay_optimization"),
        ("megatron/model/model.py",
         "整文件删除 (90行): BertModel/GPT2Model 组装器逻辑迁入 bert_model.py/gpt2_model.py"),
        ("pretrain_bert.py",
         "大规模精简 (528→?行): model_provider/train_step/evaluate 等移入 megatron/training.py"),
    ))

    mask_convention: BertMaskConvention = field(
        default_factory=BertMaskConvention
    )

    def audit(self) -> Dict[str, object]:
        """输出 BERT 模型组件的结构化报告"""
        _dbg("MANIFEST_AUDIT_START",
             f"审计 {self.upstream_commit} BertModel 组件")

        _dbg("MANIFEST_AUDIT_MASK",
             f"mask convention: {self.mask_convention.additive_masked}")
        _dbg("MANIFEST_AUDIT_HEAD",
             f"heads: {[p.name for p in BertHeadPolicy]}")

        result = {
            "upstream_commit": self.upstream_commit,
            "upstream_file": self.upstream_file,
            "upstream_lines": self.upstream_lines,
            "walpurgis_file": self.walpurgis_file,
            "components": list(self.UPSTREAM_COMPONENTS),
            "related_changes": [
                {"file": f, "change": c}
                for f, c in self.RELATED_CHANGES
            ],
            "mask_convention": self.mask_convention.as_dict(),
            "head_policies": [
                {
                    "name": p.name,
                    "uses_weight_tying": p.uses_weight_tying,
                    "input_token": p.input_token,
                    "output": p.output_description,
                }
                for p in BertHeadPolicy
            ],
        }
        _dbg("MANIFEST_AUDIT_DONE", "BertModel 审计完成 ✓")
        return result

    def self_check(self) -> None:
        """4 项断言"""
        _dbg("SELF_CHECK_START", "开始 4 项断言")

        # 1. mask convention 有效
        assert self.mask_convention.additive_masked < -1000.0
        _dbg("SELF_CHECK_1",
             f"✓ additive_masked={self.mask_convention.additive_masked}")

        # 2. MLM head 使用 weight tying
        assert BertHeadPolicy.MLM_HEAD.uses_weight_tying
        _dbg("SELF_CHECK_2", "✓ MLM_HEAD.uses_weight_tying=True")

        # 3. NSP head 不使用 weight tying
        assert not BertHeadPolicy.NSP_HEAD.uses_weight_tying
        _dbg("SELF_CHECK_3", "✓ NSP_HEAD.uses_weight_tying=False")

        # 4. 上游组件数 >= 4
        assert len(self.UPSTREAM_COMPONENTS) >= 4
        _dbg("SELF_CHECK_4",
             f"✓ {len(self.UPSTREAM_COMPONENTS)} 个上游组件")

        _dbg("SELF_CHECK_PASS", "4 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块级初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANIFEST = BertModelManifest()
_dbg("MANIFEST_INIT",
     f"BertModelManifest 初始化: {MANIFEST.upstream_lines} 行, "
     f"{len(MANIFEST.UPSTREAM_COMPONENTS)} 个组件")
MANIFEST.self_check()

_dbg("MODULE_READY", "bert_model 就绪 — 73af12903 BERT 薄包装层")

__all__ = [
    "BertMaskConvention",
    "BertHeadPolicy",
    "BertModelManifest",
    "MANIFEST",
]
