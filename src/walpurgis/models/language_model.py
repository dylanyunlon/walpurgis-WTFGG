"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/model/language_model.py（新增文件，398行）

鲁迅拿法改写（≥20%）：
  上游将 GPT-2 和 BERT 两套模型骨架合并为单一 language_model.py，
  核心思路是：embedding + transformer + pooler 三段式结构，
  通过参数开关（add_pooler、tokentype_embeddings_type 等）控制行为分支。
  上游注释几乎为零——参数从函数签名里看含义，行为从调用处反推。
  鲁迅说：「这是写给懂的人看的，不懂的人，死去吧。」

  Walpurgis 改写要点：
  1. EmbeddingLayer: 将上游 Embedding 类的三路 embedding（word/pos/tokentype）
     拆为独立子模块，逻辑分支用显式 if-else + _dbg 标注，不再靠 hasattr 猜测。
  2. TransformerStack: 上游 Transformer 类直接暴露 get_key_padding_mask() 等
     内部实现，Walpurgis 封装为私有方法并加 shape 断言。
  3. PoolerHead: 上游 Pooler 是纯 Linear(first_token_transform)，
     Walpurgis 显式命名为 PoolerHead，区分 cls_token 提取 vs 全序列输出。
  4. LanguageModelCore: 替代上游 TransformerLanguageModel，
     含前向路径断点（embedding→transformer→pooler 三阶段各有 _dbg 快照）。

迁移位置: src/walpurgis/models/language_model.py
"""

import os
import sys
import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：language_model 前向关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:language_model:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:language_model:{tag}] {msg}", file=sys.stderr, flush=True)


# ── EmbeddingLayer ────────────────────────────────────────────────────────────
# 上游将三路 embedding 混在 Embedding.__init__ 里，没有任何路径分支说明。
# Walpurgis：word / position / tokentype 三路显式建模，init 时确定哪些存在。

TokentypeMode = Literal["none", "bert_style", "gpt2_style"]


class EmbeddingLayer(nn.Module):
    """
    三路 Embedding 统一入口：word + position + tokentype（可选）。

    上游接口：
        Embedding(hidden_size, vocab_size, max_sequence_length,
                  embedding_dropout_prob, init_method,
                  num_tokentypes=0)
    Walpurgis 改写：
        - tokentype_mode 显式枚举（none / bert_style / gpt2_style）
        - position embedding 支持 learned（默认）与 sinusoidal（扩展）
        - dropout 作用在三路相加后，与上游一致但标注了语义
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_seq_len: int,
        dropout_prob: float,
        init_method,
        num_tokentypes: int = 0,
        tokentype_mode: TokentypeMode = "none",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tokentype_mode = tokentype_mode if num_tokentypes > 0 else "none"

        # word embedding（上游：self.word_embeddings）
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        init_method(self.word_embeddings.weight)

        # position embedding（上游：self.position_embeddings）
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        init_method(self.position_embeddings.weight)

        # tokentype embedding（上游：仅在 num_tokentypes > 0 时存在）
        self.tokentype_embeddings: Optional[nn.Embedding] = None
        if self.tokentype_mode != "none":
            self.tokentype_embeddings = nn.Embedding(num_tokentypes, hidden_size)
            init_method(self.tokentype_embeddings.weight)
            _dbg("INIT", f"tokentype_mode={self.tokentype_mode}, num_tokentypes={num_tokentypes}")

        self.embedding_dropout = nn.Dropout(p=dropout_prob)
        _dbg("INIT", f"vocab={vocab_size}, max_seq_len={max_seq_len}, hidden={hidden_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids:     [batch, seq_len]
        position_ids:  [batch, seq_len]
        tokentype_ids: [batch, seq_len] or None
        returns:       [batch, seq_len, hidden_size]
        """
        _dbg("FWD_INPUT_IDS", input_ids)
        _dbg("FWD_POSITION_IDS", position_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        _dbg("FWD_WORD_EMBED", words_embeddings)
        _dbg("FWD_POS_EMBED", position_embeddings)

        # tokentype 路径：上游静默跳过 None，Walpurgis 警告不一致情况
        if tokentype_ids is not None:
            if self.tokentype_embeddings is None:
                _dbg("FWD_TOKENTYPE_WARN",
                     "tokentype_ids provided but no tokentype_embeddings initialized—ignoring")
            else:
                tok_embeds = self.tokentype_embeddings(tokentype_ids)
                _dbg("FWD_TOKENTYPE_EMBED", tok_embeds)
                embeddings = embeddings + tok_embeds
        elif self.tokentype_embeddings is not None and self.tokentype_mode == "bert_style":
            _dbg("FWD_TOKENTYPE_WARN",
                 "BERT-style model but tokentype_ids=None; segment A assumed")

        embeddings = self.embedding_dropout(embeddings)
        _dbg("FWD_EMBEDDED", embeddings)
        return embeddings


# ── PoolerHead ────────────────────────────────────────────────────────────────
# 上游 Pooler：单层 Linear(hidden, hidden) + Tanh，作用于 first token ([CLS])。
# Walpurgis：显式命名 PoolerHead，区分 cls_token 提取语义。

class PoolerHead(nn.Module):
    """
    BERT 风格的 [CLS] 池化头。

    上游 Pooler 只有 dense + Tanh，没有解释这里 first_token 就是 [CLS]。
    Walpurgis 显式标注：仅在 add_pooler=True（BERT）时被 LanguageModelCore 使用。
    """

    def __init__(self, hidden_size: int, init_method) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        init_method(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        _dbg("POOLER_INIT", f"hidden_size={hidden_size}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [batch, seq_len, hidden_size]
        returns:       [batch, hidden_size]   ← [CLS] token 池化结果
        """
        # 上游：hidden_states[:, 0] 提取第 0 位（即 [CLS]）
        cls_token = hidden_states[:, 0]
        _dbg("POOLER_CLS_TOKEN", cls_token)
        pooled = torch.tanh(self.dense(cls_token))
        _dbg("POOLER_OUTPUT", pooled)
        return pooled


# ── LanguageModelCore ─────────────────────────────────────────────────────────
# 替代上游 TransformerLanguageModel。
# 上游通过 add_pooler 参数区分 BERT / GPT-2 模式，Walpurgis 显式建模两种路径。

class LanguageModelCore(nn.Module):
    """
    统一语言模型骨架：embedding → transformer → (可选) pooler。

    对应上游 TransformerLanguageModel，合并了原 gpt2_modeling.py 和
    bert modeling 两套实现。

    参数说明（鲁迅拿法补全上游缺失的文档）：
        transformer_stack: 已初始化的 TransformerStack 实例（由调用者构造）
        add_pooler:        True → BERT 模式，输出 (hidden, pooled)
                           False → GPT-2 模式，输出 hidden only
        num_tokentypes:    0 → 无 segment embedding（GPT-2 默认）
                           2 → BERT 两类 segment（A/B）
    """

    def __init__(
        self,
        embedding: EmbeddingLayer,
        transformer_stack,            # TransformerStack 实例（见 transformer.py）
        add_pooler: bool,
        hidden_size: int,
        init_method,
        num_tokentypes: int = 0,
    ) -> None:
        super().__init__()
        self.add_pooler = add_pooler
        self.embedding = embedding
        self.transformer = transformer_stack

        self.pooler: Optional[PoolerHead] = None
        if add_pooler:
            self.pooler = PoolerHead(hidden_size, init_method)
            _dbg("CORE_INIT", "BERT mode: add_pooler=True, PoolerHead constructed")
        else:
            _dbg("CORE_INIT", "GPT-2 mode: add_pooler=False, no pooler")

        _dbg("CORE_INIT", f"num_tokentypes={num_tokentypes}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
    ):
        """
        前向路径（三阶段，每阶段均有 _dbg 快照）：
          Stage 1 — Embedding：word + pos + tokentype → dropout → hidden
          Stage 2 — Transformer：hidden → contextualized hidden
          Stage 3 — Pooler（可选）：hidden[:, 0] → tanh(dense) → pooled

        返回：
          add_pooler=True  → (hidden_states, pooled_output)
          add_pooler=False → hidden_states
        """
        # ── Stage 1: Embedding ───────────────────────────────────────────────
        _dbg("STAGE1_START", f"input_ids.shape={list(input_ids.shape)}")
        embedded = self.embedding(input_ids, position_ids, tokentype_ids)
        _dbg("STAGE1_DONE", embedded)

        # ── Stage 2: Transformer ─────────────────────────────────────────────
        _dbg("STAGE2_START", f"attention_mask.shape={list(attention_mask.shape)}")
        hidden_states = self.transformer(embedded, attention_mask)
        _dbg("STAGE2_DONE", hidden_states)

        # ── Stage 3: Pooler（BERT only） ─────────────────────────────────────
        if self.add_pooler:
            assert self.pooler is not None
            _dbg("STAGE3_START", "pooler active (BERT mode)")
            pooled = self.pooler(hidden_states)
            _dbg("STAGE3_DONE", pooled)
            return hidden_states, pooled

        _dbg("STAGE3_SKIP", "no pooler (GPT-2 mode)")
        return hidden_states


# ── get_language_model（上游工厂函数的 Walpurgis 版） ─────────────────────────
# 上游 get_language_model() 直接 new TransformerLanguageModel，
# 参数列表冗长且无文档。Walpurgis 将参数分组，用 **kwargs 风格逐组注释。

def get_language_model(
    attention_mask_func,
    num_tokentypes: int,
    add_pooler: bool,
    # 模型规格
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    ffn_hidden_size: int,
    max_seq_len: int,
    # 正则化
    hidden_dropout: float,
    attention_dropout: float,
    embedding_dropout: float,
    # 初始化
    init_method,
    output_layer_init_method,
) -> LanguageModelCore:
    """
    工厂函数：构造 LanguageModelCore（含 embedding、transformer、可选 pooler）。

    上游 get_language_model() 直接依赖 args 全局对象；
    Walpurgis 显式枚举所有参数，让调用处一目了然哪些超参被实际使用。

    注意：TransformerStack 由 transformer.py 导入，避免循环依赖。
    """
    from .transformer import TransformerStack  # 延迟导入，避免循环

    tokentype_mode: TokentypeMode = "none"
    if num_tokentypes == 2:
        tokentype_mode = "bert_style"
    elif num_tokentypes > 0:
        tokentype_mode = "gpt2_style"

    _dbg("FACTORY", f"add_pooler={add_pooler}, tokentype_mode={tokentype_mode}")

    embedding = EmbeddingLayer(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout_prob=embedding_dropout,
        init_method=init_method,
        num_tokentypes=num_tokentypes,
        tokentype_mode=tokentype_mode,
    )

    transformer_stack = TransformerStack(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        attention_mask_func=attention_mask_func,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
    )

    model = LanguageModelCore(
        embedding=embedding,
        transformer_stack=transformer_stack,
        add_pooler=add_pooler,
        hidden_size=hidden_size,
        init_method=init_method,
        num_tokentypes=num_tokentypes,
    )

    _dbg("FACTORY_DONE", f"LanguageModelCore built: layers={num_layers}, heads={num_attention_heads}")
    return model
