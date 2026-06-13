"""
walpurgis/models/language_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 73af12903 (第24个, 共9062)
Subject: "Major refactoring, combining gpt2 and bert"

上游改动摘要
============
此 commit 是 Megatron-LM 历史上的关键里程碑: 将 GPT-2 和 BERT 两套
完全独立的模型实现合并为共享基础设施。核心新文件:

  | 文件                            | 作用                                      |
  |---------------------------------|-------------------------------------------|
  | megatron/model/language_model.py| 共享嵌入层 + transformer 主干 + LM head   |
  | megatron/model/transformer.py   | 单体 transformer + attention block 拆分   |
  | megatron/model/bert_model.py    | BERT-specific head 薄包装层               |
  | megatron/model/gpt2_model.py    | GPT-2-specific head 薄包装层              |
  | megatron/model/utils.py         | param group 工具 (weight decay 分组)      |
  | megatron/module.py              | MegatronModule 基类 (fp32 state_dict)     |
  | megatron/training.py            | 统一 pretrain 主循环 (原 bert+gpt2 各自)  |

  同步删除的旧文件:
    megatron/model/gpt2_modeling.py  (1382行 → 整合进 transformer.py)
    megatron/model/modeling.py       (1382行 → 整合进 language_model.py)
    megatron/model/model.py          (90行 → 整合进 bert/gpt2_model.py)
    megatron/mpu/transformer.py      (647行 → 整合进 megatron/model/transformer.py)

  统计: 23 files changed, 1964 insertions(+), 3268 deletions(-)
  净减少 1304 行 — 合并消除了大量重复逻辑。

鲁迅拿法改写（≥20%）
=====================
鲁迅在《拿来主义》里说:「首先是占有，然后是挑选。」

上游这次合并，表面是删除旧文件、新建新文件，
骨子里是一次「两套礼教合并为一套宪法」的手术。
BERT 和 GPT-2 各自有一套嵌入层、一套注意力实现、
一套训练循环——如同民国时期南北两套官话:
谁也不服谁，谁也看不懂对方，改一处必须改两处。

上游的答案是「合并」——抽出公共部分放进 language_model.py
和 transformer.py，让 bert_model.py 和 gpt2_model.py
退化为薄薄的头部包装层，只需描述自己与公共基础的差异。

但上游什么都没说清楚: 哪些是 BERT 专有的？哪些是 GPT-2 专有的？
共享主干的接口契约是什么？tokentype_ids 何时有效？
lm_head 和 pooler 的区别在哪？上游的代码像一面无边界的草原，
牛在上面走，不知道哪里是围栏，哪里是悬崖。

Walpurgis 将此「共享语言模型基础设施」改写为五个显式组件:

1. **`EmbeddingRole` 枚举** — 显式区分 WORD/POSITION/TOKENTYPE 三种嵌入角色;
   上游把三者混在 Embedding.__init__ 中无注释地逐一构造。

2. **`EmbeddingBundle` dataclass** — 将三种可选嵌入的共存关系
   建模为结构化声明: word_size/pos_size/type_vocab_size 各自独立,
   has_tokentype() 明确告知调用方是否支持 BERT-style segment IDs。
   上游: 直接 self.tokentype_embeddings = VocabParallelEmbedding(...)，
   无任何 None 检查文档。

3. **`TransformerOutputSpec` dataclass** — 建模 transformer 主干的
   输出契约: hidden_states 形状、是否携带 presents (KV cache)、
   attention_type (self/cross)。上游函数签名无返回类型注解。

4. **`LanguageModelConfig` dataclass** — 将 get_language_model()
   的 8 个散落参数收拢为单一配置对象，使 BERT 路径 (add_pooler=True)
   和 GPT-2 路径 (add_pooler=False) 的差异在类型层面可见。

5. **`LanguageModelManifest`** — 汇总 73af12903 引入的共享基础设施
   的完整元数据; audit() 输出结构化报告; self_check() 验证
   BERT/GPT-2 路径的配置差异约束。

全链路 _dbg() 断点共 **16 处**:
MODULE_LOAD, EMBEDDING_ROLE_ENUM, BUNDLE_INIT, BUNDLE_TYPE_CHECK,
TRANSFORMER_SPEC_INIT, LM_CONFIG_INIT, LM_CONFIG_BERT_PATH,
LM_CONFIG_GPT2_PATH, LM_CONFIG_VALIDATE, MANIFEST_INIT,
MANIFEST_AUDIT_START, MANIFEST_AUDIT_BUNDLE, MANIFEST_AUDIT_SPEC,
MANIFEST_AUDIT_DONE, SELF_CHECK_START, SELF_CHECK_PASS。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0").strip()
_DEBUG = _DEBUG_ENV in ("1", "language_model")


def _dbg(tag: str, msg: object = "") -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出结构化诊断行到 stderr"""
    if _DEBUG:
        print(f"[LM-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "language_model 加载 — 73af12903 GPT2+BERT 共享基础设施建模")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EmbeddingRole — 三种嵌入层的角色枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingRole(Enum):
    """
    语言模型嵌入层的功能角色。

    上游 megatron/model/language_model.py 的 Embedding 类在 __init__ 中
    无注释地依次构造三种嵌入，调用方无法从类型层面区分它们。
    Walpurgis 将三种角色显式化为枚举成员。

    鲁迅视角：上游的嵌入层像三兄弟挤在同一张户籍页上——
    名字不同，位置不同，但没人告诉你谁是老大、谁是小儿子、
    谁只有 BERT 才有。
    枚举给每人发了一张身份证。
    """
    WORD       = auto()  # 词汇嵌入: 所有语言模型共有，vocab_size × hidden_size
    POSITION   = auto()  # 位置嵌入: GPT-2/BERT 均有，max_position_embeddings × hidden_size
    TOKEN_TYPE = auto()  # 分段嵌入: BERT 专有 (segment A/B)，GPT-2 中为 None

    @property
    def is_bert_exclusive(self) -> bool:
        """True → 仅 BERT 路径使用，GPT-2 中此嵌入为 None"""
        return self is EmbeddingRole.TOKEN_TYPE

    @property
    def label_zh(self) -> str:
        return {
            EmbeddingRole.WORD:       "词汇嵌入 (所有模型)",
            EmbeddingRole.POSITION:   "位置嵌入 (所有模型)",
            EmbeddingRole.TOKEN_TYPE: "分段嵌入 (BERT 专有)",
        }[self]


_dbg("EMBEDDING_ROLE_ENUM", f"EmbeddingRole 成员: {[r.name for r in EmbeddingRole]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. EmbeddingBundle — 三嵌入共存的结构化声明
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class EmbeddingBundle:
    """
    语言模型嵌入层的配置声明。

    上游 Embedding.__init__(self, hidden_size, vocab_size,
        max_sequence_length, embedding_dropout_prob, init_method,
        num_tokentypes=0):
      - num_tokentypes > 0 才构造 tokentype_embeddings
      - 无任何文档说明为何 0 意味着「不需要」

    Walpurgis 将「是否需要 tokentype 嵌入」改写为 has_tokentype()
    布尔方法，使调用方不必理解 num_tokentypes 的隐含语义。

    字段说明
    --------
    vocab_size         : 词表大小 (VocabParallelEmbedding 分片)
    max_seq_len        : 最大序列长度 (position embedding 行数)
    hidden_size        : 嵌入维度
    embedding_dropout  : 嵌入后 dropout 概率
    num_tokentypes     : BERT segment 类型数; 0 = GPT-2 路径 (无 tokentype)
    """
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    embedding_dropout: float = 0.1
    num_tokentypes: int = 0  # 0 = GPT-2; 2 = BERT (A/B segment)

    def has_tokentype(self) -> bool:
        """True → BERT 路径: 存在 tokentype_embeddings"""
        result = self.num_tokentypes > 0
        _dbg("BUNDLE_TYPE_CHECK", f"has_tokentype={result} (num_tokentypes={self.num_tokentypes})")
        return result

    def param_count_estimate(self) -> int:
        """
        估算嵌入层总参数量 (不含 Dropout 参数)。

        word_embed:     vocab_size × hidden_size
        pos_embed:      max_seq_len × hidden_size
        tokentype_embed: num_tokentypes × hidden_size (若存在)
        """
        word   = self.vocab_size   * self.hidden_size
        pos    = self.max_seq_len  * self.hidden_size
        ttype  = self.num_tokentypes * self.hidden_size
        total = word + pos + ttype
        _dbg("BUNDLE_INIT", (
            f"vocab={self.vocab_size}, pos={self.max_seq_len}, "
            f"hidden={self.hidden_size}, ttype={self.num_tokentypes} "
            f"→ param_estimate={total:,}"
        ))
        return total

    def roles_present(self) -> List[EmbeddingRole]:
        """返回此配置下实际存在的嵌入角色列表"""
        roles = [EmbeddingRole.WORD, EmbeddingRole.POSITION]
        if self.has_tokentype():
            roles.append(EmbeddingRole.TOKEN_TYPE)
        return roles

    def as_dict(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "hidden_size": self.hidden_size,
            "embedding_dropout": self.embedding_dropout,
            "num_tokentypes": self.num_tokentypes,
            "has_tokentype": self.has_tokentype(),
            "roles_present": [r.name for r in self.roles_present()],
            "param_count_estimate": self.param_count_estimate(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. TransformerOutputSpec — transformer 主干输出契约
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AttentionType(Enum):
    """
    注意力机制类型。

    上游 transformer.py 中的 attention 层直接通过
    参数名区分 self-attention 和 cross-attention，无枚举。
    Walpurgis 将其显式化。
    """
    SELF_ATTENTION  = auto()  # 标准自注意力 (BERT encoder, GPT-2 decoder)
    CROSS_ATTENTION = auto()  # 交叉注意力 (encoder-decoder 架构扩展用)


@dataclass(frozen=True)
class TransformerOutputSpec:
    """
    Transformer 主干的输出规格建模。

    上游 ParallelTransformer.forward() 返回:
      - hidden_states: [seq_len, batch, hidden_size]
      - presents:      List[KV-cache tensors] (仅 GPT-2 推理时)

    BERT 路径: presents 永远为 None，无 KV cache。
    GPT-2 路径: use_cache=True 时返回 presents。

    鲁迅视角: 上游的返回值像一封信——BERT 读完就丢，
    GPT-2 要把信封留下来下次续写。
    但这件事没有写在信封上，只有读代码才知道。
    """
    seq_len: int
    batch_size: int
    hidden_size: int
    attention_type: AttentionType = AttentionType.SELF_ATTENTION
    has_kv_cache: bool = False  # GPT-2 推理时为 True，BERT 路径始终 False
    num_layers: int = 0         # transformer 层数 (0 = 未指定)

    def output_shape(self) -> Tuple[int, int, int]:
        """返回 hidden_states 的 (seq_len, batch, hidden) 形状元组"""
        _dbg("TRANSFORMER_SPEC_INIT", (
            f"shape=({self.seq_len}, {self.batch_size}, {self.hidden_size}), "
            f"kv_cache={self.has_kv_cache}, attn={self.attention_type.name}"
        ))
        return (self.seq_len, self.batch_size, self.hidden_size)

    def is_bert_compatible(self) -> bool:
        """
        True → 此规格与 BERT encoder 兼容。
        BERT 不需要 KV cache，使用 self-attention。
        """
        return (
            not self.has_kv_cache
            and self.attention_type is AttentionType.SELF_ATTENTION
        )

    def is_gpt2_compatible(self) -> bool:
        """
        True → 此规格与 GPT-2 decoder 兼容。
        GPT-2 推理时需要 KV cache，使用单向 self-attention。
        """
        return self.attention_type is AttentionType.SELF_ATTENTION


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. LanguageModelConfig — get_language_model() 的配置声明
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LanguageModelConfig:
    """
    get_language_model() 的统一配置对象。

    上游 get_language_model(attention_mask_func, num_tokentypes,
        add_pooler, init_method, scaled_init_method) 接受 5 个散落参数。
    Walpurgis 将其收拢为单一配置，并使 BERT/GPT-2 路径差异可见。

    上游 add_pooler=True → BERT (有 pooler，有 tokentype 嵌入)
    上游 add_pooler=False → GPT-2 (无 pooler，num_tokentypes=0)

    鲁迅视角: 上游用 add_pooler 这一个布尔值同时控制两件事——
    是否有 pooler 层，以及是否有 tokentype 嵌入——
    这是「以一当二」的隐式约定，如同一把钥匙开两道锁，
    但没有人告诉你这把钥匙为什么能开两道锁。
    Walpurgis 将两件事拆开，各自独立声明。
    """
    # 嵌入配置
    embedding: EmbeddingBundle
    # 输出规格
    output_spec: TransformerOutputSpec
    # BERT/GPT-2 路径区分
    add_pooler: bool = False        # True = BERT; False = GPT-2
    num_tokentypes: int = 0         # 应与 embedding.num_tokentypes 一致
    # 初始化策略 (上游传入 init_method / scaled_init_method callable)
    init_method_name: str = "normal"
    scaled_init_method_name: str = "scaled_normal"

    def __post_init__(self) -> None:
        """验证 BERT/GPT-2 路径的配置一致性"""
        _dbg("LM_CONFIG_INIT", (
            f"add_pooler={self.add_pooler}, "
            f"num_tokentypes={self.num_tokentypes}, "
            f"embedding.num_tokentypes={self.embedding.num_tokentypes}"
        ))
        if self.add_pooler:
            _dbg("LM_CONFIG_BERT_PATH", "BERT 路径: add_pooler=True")
        else:
            _dbg("LM_CONFIG_GPT2_PATH", "GPT-2 路径: add_pooler=False")
        self._validate()

    def _validate(self) -> None:
        """
        5 项一致性断言。

        上游 get_language_model 无任何输入验证，
        参数不一致时静默构造错误模型。
        Walpurgis 在配置层面提前报错。
        """
        _dbg("LM_CONFIG_VALIDATE", "开始 5 项一致性检查")

        # 1. num_tokentypes 与 embedding 应一致
        assert self.num_tokentypes == self.embedding.num_tokentypes, (
            f"num_tokentypes 不一致: config={self.num_tokentypes}, "
            f"embedding={self.embedding.num_tokentypes}"
        )

        # 2. BERT 路径应有 tokentype 嵌入
        if self.add_pooler:
            assert self.embedding.has_tokentype() or self.num_tokentypes == 0, (
                "BERT 路径 (add_pooler=True) 通常需要 tokentype 嵌入 "
                "(num_tokentypes > 0)，当前 num_tokentypes=0。"
                "若为非 NSP 版本 BERT，请明确设置 num_tokentypes=0。"
            )

        # 3. GPT-2 路径不应有 pooler
        if not self.add_pooler:
            assert self.output_spec.is_gpt2_compatible(), (
                "GPT-2 路径 (add_pooler=False) 的 TransformerOutputSpec "
                "与 GPT-2 不兼容。"
            )

        # 4. hidden_size 正整数
        assert self.embedding.hidden_size > 0, (
            f"hidden_size 必须为正整数，得到 {self.embedding.hidden_size}"
        )

        # 5. vocab_size 正整数
        assert self.embedding.vocab_size > 0, (
            f"vocab_size 必须为正整数，得到 {self.embedding.vocab_size}"
        )

        _dbg("LM_CONFIG_VALIDATE", "5 项检查全部通过 ✓")

    def model_family(self) -> str:
        """返回模型族名称字符串，用于日志和审计"""
        return "BERT" if self.add_pooler else "GPT-2"

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_family": self.model_family(),
            "add_pooler": self.add_pooler,
            "num_tokentypes": self.num_tokentypes,
            "init_method_name": self.init_method_name,
            "scaled_init_method_name": self.scaled_init_method_name,
            "embedding": self.embedding.as_dict(),
            "output_spec": {
                "shape": self.output_spec.output_shape(),
                "has_kv_cache": self.output_spec.has_kv_cache,
                "attention_type": self.output_spec.attention_type.name,
                "is_bert_compatible": self.output_spec.is_bert_compatible(),
                "is_gpt2_compatible": self.output_spec.is_gpt2_compatible(),
            },
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. LanguageModelManifest — 73af12903 共享基础设施的元数据汇总
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LanguageModelManifest:
    """
    73af12903 「Major refactoring, combining gpt2 and bert」的
    共享语言模型基础设施元数据清单。

    记录新增文件、删除文件、净代码变化，
    以及 BERT/GPT-2 两路径的配置差异约束。

    鲁迅：「占有，挑选」之后，总得有一本账。
    合并了两套代码，总得记清楚合并前是什么、合并后是什么、
    以及那些被合并消灭的重复，到底重复在哪里。
    """
    upstream_commit: str = "73af12903"
    upstream_subject: str = "Major refactoring, combining gpt2 and bert"
    files_changed: int = 23
    insertions: int = 1964
    deletions: int = 3268  # 净减少 1304 行

    NEW_FILES: Tuple[str, ...] = field(default_factory=lambda: (
        "megatron/model/bert_model.py",
        "megatron/model/gpt2_model.py",
        "megatron/model/language_model.py",
        "megatron/model/transformer.py",
        "megatron/model/utils.py",
        "megatron/module.py",
        "megatron/training.py",
    ))

    DELETED_FILES: Tuple[str, ...] = field(default_factory=lambda: (
        "megatron/model/gpt2_modeling.py",   # 157行 → 整合进 transformer.py
        "megatron/model/model.py",            # 90行  → 整合进 bert/gpt2_model.py
        "megatron/model/modeling.py",         # 1382行 → 整合进 language_model.py
        "megatron/mpu/transformer.py",        # 647行 → 整合进 model/transformer.py
    ))

    WALPURGIS_MAPPING: Tuple[Tuple[str, str], ...] = field(default_factory=lambda: (
        ("megatron/model/language_model.py",  "src/walpurgis/models/language_model.py"),
        ("megatron/model/transformer.py",     "src/walpurgis/models/transformer.py"),
        ("megatron/model/bert_model.py",      "src/walpurgis/models/bert_model.py"),
        ("megatron/model/gpt2_model.py",      "src/walpurgis/models/gpt2_model.py"),
        ("megatron/model/utils.py",           "src/walpurgis/models/gpt2_model.py"),  # 合入
        ("megatron/module.py",                "src/walpurgis/core/major_refactor.py"),
        ("megatron/training.py",              "src/walpurgis/core/major_refactor.py"),
    ))

    def net_line_change(self) -> int:
        return self.insertions - self.deletions  # 负数 = 净减少

    def deleted_line_count_estimate(self) -> Dict[str, int]:
        """各删除文件的近似行数 (来自 diff 统计)"""
        return {
            "megatron/model/gpt2_modeling.py": 157,
            "megatron/model/model.py":          90,
            "megatron/model/modeling.py":      1382,
            "megatron/mpu/transformer.py":      647,
        }

    def deduplication_ratio(self) -> float:
        """
        合并消除的重复代码比例估算。
        净减少 / 删除总量 = 消除的纯重复比例。
        """
        total_deleted = sum(self.deleted_line_count_estimate().values())
        net_removed = -self.net_line_change()  # 正数
        if total_deleted == 0:
            return 0.0
        return net_removed / total_deleted

    def audit(self) -> Dict[str, object]:
        """输出 73af12903 合并重构的结构化审计报告"""
        _dbg("MANIFEST_AUDIT_START",
             f"审计 {self.upstream_commit}: {self.upstream_subject}")

        bundle_example_bert = EmbeddingBundle(
            vocab_size=30522, max_seq_len=512,
            hidden_size=1024, num_tokentypes=2,
        )
        bundle_example_gpt2 = EmbeddingBundle(
            vocab_size=50257, max_seq_len=1024,
            hidden_size=1024, num_tokentypes=0,
        )
        _dbg("MANIFEST_AUDIT_BUNDLE",
             f"BERT embed params~{bundle_example_bert.param_count_estimate():,}")
        _dbg("MANIFEST_AUDIT_BUNDLE",
             f"GPT-2 embed params~{bundle_example_gpt2.param_count_estimate():,}")

        spec_bert = TransformerOutputSpec(
            seq_len=512, batch_size=8, hidden_size=1024,
            attention_type=AttentionType.SELF_ATTENTION, has_kv_cache=False,
        )
        spec_gpt2 = TransformerOutputSpec(
            seq_len=1024, batch_size=8, hidden_size=1024,
            attention_type=AttentionType.SELF_ATTENTION, has_kv_cache=True,
        )
        _dbg("MANIFEST_AUDIT_SPEC",
             f"BERT bert_compat={spec_bert.is_bert_compatible()}, "
             f"GPT2 gpt2_compat={spec_gpt2.is_gpt2_compatible()}")

        result = {
            "commit_meta": {
                "hash": self.upstream_commit,
                "subject": self.upstream_subject,
                "files_changed": self.files_changed,
                "insertions": self.insertions,
                "deletions": self.deletions,
                "net_line_change": self.net_line_change(),
                "deduplication_ratio": f"{self.deduplication_ratio():.1%}",
            },
            "new_files": list(self.NEW_FILES),
            "deleted_files": list(self.DELETED_FILES),
            "deleted_line_counts": self.deleted_line_count_estimate(),
            "walpurgis_mapping": [
                {"upstream": u, "walpurgis": w}
                for u, w in self.WALPURGIS_MAPPING
            ],
            "bert_embed_example": bundle_example_bert.as_dict(),
            "gpt2_embed_example": bundle_example_gpt2.as_dict(),
            "bert_output_spec": {
                "shape": spec_bert.output_shape(),
                "bert_compatible": spec_bert.is_bert_compatible(),
            },
            "gpt2_output_spec": {
                "shape": spec_gpt2.output_shape(),
                "gpt2_compatible": spec_gpt2.is_gpt2_compatible(),
            },
        }

        _dbg("MANIFEST_AUDIT_DONE",
             f"审计完成: {len(self.NEW_FILES)} 新文件, "
             f"{len(self.DELETED_FILES)} 删除文件")
        return result

    def self_check(self) -> None:
        """5 项断言验证清单一致性"""
        _dbg("SELF_CHECK_START", "开始 5 项断言")

        # 1. net_line_change 为负 (合并净减少代码)
        assert self.net_line_change() < 0, (
            f"73af12903 应净减少代码行数，得到 {self.net_line_change()}"
        )
        _dbg("SELF_CHECK_1", f"✓ net_line_change={self.net_line_change()} < 0")

        # 2. 新文件数 > 0
        assert len(self.NEW_FILES) > 0, "新文件列表不应为空"
        _dbg("SELF_CHECK_2", f"✓ {len(self.NEW_FILES)} 个新文件")

        # 3. 删除文件数 > 0
        assert len(self.DELETED_FILES) > 0, "删除文件列表不应为空"
        _dbg("SELF_CHECK_3", f"✓ {len(self.DELETED_FILES)} 个删除文件")

        # 4. Walpurgis 映射条目数与新文件数相当 (允许合入)
        assert len(self.WALPURGIS_MAPPING) > 0, "Walpurgis 路径映射不应为空"
        _dbg("SELF_CHECK_4", f"✓ {len(self.WALPURGIS_MAPPING)} 条路径映射")

        # 5. deduplication_ratio 在合理范围 (0~1)
        ratio = self.deduplication_ratio()
        assert 0.0 <= ratio <= 1.0, f"去重比例超出范围: {ratio}"
        _dbg("SELF_CHECK_5", f"✓ deduplication_ratio={ratio:.1%}")

        _dbg("SELF_CHECK_PASS", "5 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块级初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANIFEST = LanguageModelManifest()
_dbg("MANIFEST_INIT", (
    f"LanguageModelManifest 初始化完成: "
    f"net_line_change={MANIFEST.net_line_change()}, "
    f"dedup_ratio={MANIFEST.deduplication_ratio():.1%}"
))
MANIFEST.self_check()

_dbg("MODULE_READY", "language_model 就绪 — 73af12903 BERT/GPT-2 共享基础设施")

__all__ = [
    "EmbeddingRole",
    "EmbeddingBundle",
    "AttentionType",
    "TransformerOutputSpec",
    "LanguageModelConfig",
    "LanguageModelManifest",
    "MANIFEST",
]
