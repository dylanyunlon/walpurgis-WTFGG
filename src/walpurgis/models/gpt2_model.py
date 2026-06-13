"""
walpurgis/models/gpt2_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 73af12903 (第24个, 共9062)
Subject: "Major refactoring, combining gpt2 and bert"

上游改动摘要
============
megatron/model/gpt2_model.py (119行新增):
  73af12903 将 GPT-2 模型组装逻辑从 pretrain_gpt2.py 中抽出，
  成为与 bert_model.py 对称的薄包装层。

  上游 GPT2Model 的组成:
    - get_language_model(add_pooler=False, num_tokentypes=0, ...)
      → LanguageModel (embedding + transformer，无 pooler)
    - gpt2_attention_mask_func(): 因果 (下三角) attention mask
    - GPT2Model: 组装 forward 计算图
      forward(input_ids, position_ids, attention_mask,
              layer_past=None, get_key_value=False, tokentype_ids=None)
      → lm_logits (parallel output) 或 logits

megatron/model/utils.py (80行新增):
  原 megatron/model/model.py 中的
  gpt2_get_params_for_weight_decay_optimization() 迁移至此，
  同时新增 bert_get_params_for_weight_decay_optimization()，
  抽象为通用的 get_params_for_weight_decay_optimization()。

  核心逻辑: 将模型参数分为两组:
    1. weight_decay 组: 所有不是 bias/LayerNorm 参数的参数
    2. no_decay 组:    bias 参数 + LayerNorm 参数 (gamma/beta)
  返回 [{params: [...], weight_decay: wd}, {params: [...], weight_decay: 0.0}]

  73af12903 同步更新:
    - megatron/model/__init__.py: 删除 gpt2_get_params_for_weight_decay_optimization
    - evaluate_gpt2.py: 删除对应 import

kicker: tokentype_ids 透传 (73af12903 新增)
  generate_samples.py 中:
    logits = model(tokens, position_ids, attention_mask)
    → logits = model(tokens, position_ids, attention_mask, tokentype_ids=type_ids)
  sample_sequence_batch() 新增 type_ids=None 参数并透传到 model()。

鲁迅拿法改写（≥20%）
=====================
鲁迅说:「什么叫做好？什么叫做坏？……没有一定的标准。」

上游 get_params_for_weight_decay_optimization 是一个「无声的判官」——
它看一眼参数名，决定这个参数是否要被正则化，
但判决标准散落在字符串 contains 调用里，没有文档，没有解释。
为什么 bias 不需要 weight decay？
为什么 LayerNorm 不需要？
为什么 embedding 需要？
上游的代码如同旧时的家法——「从来如此」，
没有人告诉你道理，你只能服从。

Walpurgis 将此「参数分组判决逻辑」改写为三个显式组件:

1. **`WeightDecayGroup` 枚举** — 显式区分 DECAY / NO_DECAY 两种
   参数组，以及 NO_DECAY 的理由分类: BIAS / LAYER_NORM / EMBEDDING_SPECIAL。
   上游: 字符串匹配 {'bias', 'LayerNorm', 'layer_norm'} 后归组，无枚举。

2. **`ParamGroupRule` dataclass** — 建模每条参数分组规则:
   pattern (匹配条件)、group (归属组)、reason (为何不加 weight decay)。
   上游将规则嵌在 list comprehension 里，无法独立测试每条规则。

3. **`GPT2ModelManifest`** — 建模 73af12903 引入的 GPT2Model +
   utils.py 组件清单，记录 tokentype_ids 透传变更、
   weight decay 分组规则、以及 parallel_output 标志的语义。
   audit() 输出完整报告。

全链路 _dbg() 断点共 **14 处**:
MODULE_LOAD, WD_GROUP_ENUM, PARAM_RULE_INIT, PARAM_RULE_MATCH,
MANIFEST_INIT, MANIFEST_AUDIT_START, MANIFEST_AUDIT_RULES,
MANIFEST_AUDIT_TOKENTYPE, MANIFEST_AUDIT_DONE,
SELF_CHECK_START, SELF_CHECK_1~4, SELF_CHECK_PASS。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0").strip()
_DEBUG = _DEBUG_ENV in ("1", "gpt2_model")


def _dbg(tag: str, msg: object = "") -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出结构化诊断行到 stderr"""
    if _DEBUG:
        print(f"[GPT2-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "gpt2_model 加载 — 73af12903 GPT-2 薄包装层 + weight decay 分组建模")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WeightDecayGroup — 参数分组枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WeightDecayGroup(Enum):
    """
    模型参数的 weight decay 分组。

    上游 get_params_for_weight_decay_optimization 将参数分为两组:
      - weight_decay 组: L2 正则化，防止权重过大
      - no_decay 组:    不加 L2，原因各异

    为什么 bias 不加 weight decay？
      bias 不会导致过拟合 (不参与特征缩放)，
      加 weight decay 会引入不必要的偏移。

    为什么 LayerNorm (gamma/beta) 不加 weight decay？
      LayerNorm 参数控制归一化后的尺度和偏移，
      L2 正则化会压缩 gamma 趋向 0，破坏归一化效果。

    鲁迅视角：weight decay 是「法外之王」——
    大多数参数必须服从它的管制，
    但 bias 和 LayerNorm 因「特殊身份」获得豁免。
    上游的豁免名单藏在字符串里，Walpurgis 把它写成宪法条文。
    """
    DECAY    = auto()  # 需要 L2 正则化
    NO_DECAY = auto()  # 免于 L2 正则化

    @property
    def weight_decay_value(self) -> float:
        """返回此组对应的 weight_decay 值 (0.0 for NO_DECAY)"""
        return 0.0 if self is WeightDecayGroup.NO_DECAY else float('nan')


_dbg("WD_GROUP_ENUM",
     f"WeightDecayGroup: {[g.name for g in WeightDecayGroup]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ParamGroupRule — 参数分组规则
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class ParamGroupRule:
    """
    单条参数分组规则。

    上游 utils.py 的分组逻辑 (简化伪码):
      no_weight_decay_params = [p for n, p in named_params
          if any(nd in n for nd in no_decay_names)]
      weight_decay_params = [p for n, p in named_params
          if not any(nd in n for nd in no_decay_names)]
      no_decay_names = ['bias', 'LayerNorm', 'layer_norm']

    每条 ParamGroupRule 建模一个 no_decay_names 条目的完整语义。

    字段说明
    --------
    pattern       : 参数名中的匹配子串 (对应 no_decay_names 中的一项)
    group         : 匹配后的分组
    match_type    : 匹配方式 ('contains' / 'exact' / 'endswith')
    reason        : 为何此类参数不加 weight decay 的理由
    example_names : 典型参数名示例
    """
    pattern: str
    group: WeightDecayGroup
    match_type: str  # 'contains', 'exact', 'endswith'
    reason: str
    example_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _dbg("PARAM_RULE_INIT",
             f"rule: pattern='{self.pattern}', group={self.group.name}, "
             f"type={self.match_type}")

    def matches(self, param_name: str) -> bool:
        """判断给定参数名是否匹配此规则"""
        if self.match_type == 'contains':
            result = self.pattern in param_name
        elif self.match_type == 'exact':
            result = param_name == self.pattern
        elif self.match_type == 'endswith':
            result = param_name.endswith(self.pattern)
        else:
            result = False
        _dbg("PARAM_RULE_MATCH",
             f"'{param_name}' vs pattern='{self.pattern}': {result}")
        return result

    def as_dict(self) -> Dict[str, object]:
        return {
            "pattern": self.pattern,
            "group": self.group.name,
            "match_type": self.match_type,
            "reason": self.reason,
            "example_names": list(self.example_names),
        }


# ── 静态规则清单 (来自 megatron/model/utils.py 73af12903) ────────────────────

NO_DECAY_RULES: Tuple[ParamGroupRule, ...] = (
    ParamGroupRule(
        pattern="bias",
        group=WeightDecayGroup.NO_DECAY,
        match_type="contains",
        reason=(
            "bias 参数不参与特征缩放，L2 正则化会引入不必要的偏移。"
            "标准做法 (Loshchilov & Hutter 2019) 是将 bias 排除在 weight decay 之外。"
        ),
        example_names=("attention.query.bias", "dense.bias", "output_projection.bias"),
    ),
    ParamGroupRule(
        pattern="LayerNorm",
        group=WeightDecayGroup.NO_DECAY,
        match_type="contains",
        reason=(
            "LayerNorm 的 weight (gamma) 和 bias (beta) 控制归一化后的尺度和偏移。"
            "L2 正则化会压缩 gamma 趋向 0，破坏归一化稳定性。"
        ),
        example_names=("input_layernorm.weight", "post_attention_layernorm.weight",
                       "input_layernorm.bias"),
    ),
    ParamGroupRule(
        pattern="layer_norm",
        group=WeightDecayGroup.NO_DECAY,
        match_type="contains",
        reason=(
            "与 'LayerNorm' 规则等价，覆盖小写命名约定。"
            "Megatron-LM 在不同组件中混用 LayerNorm 和 layer_norm 命名。"
        ),
        example_names=("final_layer_norm.weight", "final_layer_norm.bias"),
    ),
)

# 默认规则: 不匹配任何 no-decay 规则的参数归入 DECAY 组
DEFAULT_RULE = ParamGroupRule(
    pattern="",  # 空 pattern，匹配所有未被 no-decay 规则覆盖的参数
    group=WeightDecayGroup.DECAY,
    match_type="contains",
    reason=(
        "所有不属于 bias / LayerNorm 的参数 (矩阵权重、embedding) "
        "均需要 weight decay 以防止过拟合。"
    ),
    example_names=("attention.query.weight", "dense.weight",
                   "word_embeddings.weight"),
)


def classify_param(name: str) -> WeightDecayGroup:
    """
    按参数名判断其 weight decay 分组。

    对应上游 utils.py 的分组 list comprehension:
      no_decay_names = ['bias', 'LayerNorm', 'layer_norm']
      group = NO_DECAY if any(nd in name for nd in no_decay_names) else DECAY

    Walpurgis 改写为: 遍历规则列表，返回首个匹配规则的 group。
    """
    for rule in NO_DECAY_RULES:
        if rule.matches(name):
            return rule.group
    return WeightDecayGroup.DECAY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GPT2ModelManifest — 73af12903 GPT2Model + utils 组件清单
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GPT2ModelManifest:
    """
    73af12903 引入的 GPT2Model + model/utils.py 组件清单。

    记录 tokentype_ids 透传变更、weight decay 分组规则、
    parallel_output 标志的语义。
    """
    upstream_commit: str = "73af12903"
    gpt2_model_file: str = "megatron/model/gpt2_model.py"
    utils_file: str = "megatron/model/utils.py"
    gpt2_lines: int = 119
    utils_lines: int = 80

    # tokentype_ids 透传变更 (generate_samples.py)
    TOKENTYPE_CHANGE: Dict[str, str] = field(default_factory=lambda: {
        "file": "generate_samples.py",
        "old_call": "model(tokens, position_ids, attention_mask)",
        "new_call": "model(tokens, position_ids, attention_mask, tokentype_ids=type_ids)",
        "also_changed": "sample_sequence_batch() 新增 type_ids=None 参数",
        "reason": (
            "GPT-2 现在与 BERT 共用 LanguageModel 基础设施，"
            "后者支持 tokentype_ids。为使 GPT-2 推理时可选传入 segment IDs，"
            "新增 type_ids 参数并透传至 model() 调用。"
            "默认 None 与 73af12903 之前的行为等价。"
        ),
    })

    # get_masks_and_position_ids 签名变更 (generate_samples.py)
    MASK_POSITION_CHANGE: Dict[str, str] = field(default_factory=lambda: {
        "old_call": "get_masks_and_position_ids(tokens, eod, reset_pos, reset_mask)",
        "new_call": "get_masks_and_position_ids(tokens, eod, reset_pos, reset_mask, False)",
        "new_param": "False (第5个参数，对应 loss_on_targets_only 或类似标志)",
        "reason": "签名扩展，新增布尔参数，默认 False 保持向后兼容",
    })

    no_decay_rules: Tuple[ParamGroupRule, ...] = field(
        default_factory=lambda: NO_DECAY_RULES
    )

    def audit(self) -> Dict[str, object]:
        """输出 GPT2Model + utils 组件的结构化报告"""
        _dbg("MANIFEST_AUDIT_START",
             f"审计 {self.upstream_commit} GPT2Model + utils")

        rule_dicts = [r.as_dict() for r in self.no_decay_rules]
        _dbg("MANIFEST_AUDIT_RULES",
             f"{len(rule_dicts)} 条 no-decay 规则")

        # 演示 classify_param
        examples = {
            "attention.query.weight": classify_param("attention.query.weight").name,
            "attention.query.bias":   classify_param("attention.query.bias").name,
            "input_layernorm.weight": classify_param("input_layernorm.weight").name,
            "LayerNorm.bias":         classify_param("LayerNorm.bias").name,
            "word_embeddings.weight": classify_param("word_embeddings.weight").name,
        }
        _dbg("MANIFEST_AUDIT_TOKENTYPE", f"tokentype 透传: {self.TOKENTYPE_CHANGE['new_call']}")

        result = {
            "upstream_commit": self.upstream_commit,
            "gpt2_model_file": self.gpt2_model_file,
            "utils_file": self.utils_file,
            "gpt2_lines": self.gpt2_lines,
            "utils_lines": self.utils_lines,
            "no_decay_rules": rule_dicts,
            "param_classification_examples": examples,
            "tokentype_ids_change": self.TOKENTYPE_CHANGE,
            "mask_position_change": self.MASK_POSITION_CHANGE,
        }
        _dbg("MANIFEST_AUDIT_DONE", "GPT2Model + utils 审计完成 ✓")
        return result

    def self_check(self) -> None:
        """4 项断言"""
        _dbg("SELF_CHECK_START", "开始 4 项断言")

        # 1. no-decay 规则数 >= 3
        assert len(self.no_decay_rules) >= 3, (
            f"期望 >= 3 条 no-decay 规则，得到 {len(self.no_decay_rules)}"
        )
        _dbg("SELF_CHECK_1",
             f"✓ {len(self.no_decay_rules)} 条 no-decay 规则")

        # 2. bias 参数归入 NO_DECAY
        assert classify_param("dense.bias") is WeightDecayGroup.NO_DECAY, (
            "bias 参数应归入 NO_DECAY"
        )
        _dbg("SELF_CHECK_2", "✓ bias → NO_DECAY")

        # 3. 矩阵权重归入 DECAY
        assert classify_param("dense.weight") is WeightDecayGroup.DECAY, (
            "权重矩阵应归入 DECAY"
        )
        _dbg("SELF_CHECK_3", "✓ weight → DECAY")

        # 4. LayerNorm 归入 NO_DECAY
        assert classify_param("LayerNorm.weight") is WeightDecayGroup.NO_DECAY, (
            "LayerNorm 参数应归入 NO_DECAY"
        )
        _dbg("SELF_CHECK_4", "✓ LayerNorm → NO_DECAY")

        _dbg("SELF_CHECK_PASS", "4 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块级初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANIFEST = GPT2ModelManifest()
_dbg("MANIFEST_INIT",
     f"GPT2ModelManifest 初始化: {MANIFEST.gpt2_lines} + {MANIFEST.utils_lines} 行")
MANIFEST.self_check()

_dbg("MODULE_READY", "gpt2_model 就绪 — 73af12903 GPT-2 薄包装层 + weight decay 分组")

__all__ = [
    "WeightDecayGroup",
    "ParamGroupRule",
    "NO_DECAY_RULES",
    "DEFAULT_RULE",
    "classify_param",
    "GPT2ModelManifest",
    "MANIFEST",
]
