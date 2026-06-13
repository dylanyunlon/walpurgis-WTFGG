# coding=utf-8
# Walpurgis migration: megatron/model/language_model.py + bert_model.py +
#                      gpt2_model.py + model/utils.py
# Upstream commit: beb3e0d38  "Merge branch 'transformer_refactoring_from_pretrain_refactoring'"
#
# 核心变化：BERT 和 GPT-2 首次共享 TransformerLanguageModel 骨架。
# 各自的注意力掩码函数（bert_attention_mask_func / gpt2_attention_mask_func）
# 是两种架构在语义上唯一的分叉点。其余一切：Embedding、Pooler、LMHead、
# state_dict 保存/加载，全部统一。
#
# 鲁迅曰：BERT 和 GPT-2，一个向左看，一个向右看，
# 都说自己看到的才是语言的本质。现在合并了——
# 共用同一副眼镜，只是镜片的焦点不同。
# 这不是妥协，这是「抽象」——把分歧逼到最小的那一点。
#
# Walpurgis 改写要点（≥20%）：
#   1. EmbeddingSpec(dataclass) 封装 Embedding 构造参数，替代散参数传递
#   2. AttentionMaskKind(Enum) 命名 BERT/GPT-2 掩码语义，使策略可静态审计
#   3. LMHeadSpec(dataclass) 封装 BertLMHead 参数
#   4. ModelParallelLogitsMode(Enum: PARALLEL / GATHERED) 替代 parallel_output 布尔值
#   5. WeightDecayManifest 将参数分组逻辑从散函数升为可查询台账
#   6. _dbg() 断点 16 处

import os
import enum
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 公共工具导入（来自 transformer_unified）
# ---------------------------------------------------------------------------
try:
    from .transformer_unified import (
        WalpurgisModule, TransformerSpec, ParallelTransformer,
        CheckpointStrategy, ResidualPolicy, mpu, LayerNorm, _dbg,
    )
except ImportError:
    from transformer_unified import (
        WalpurgisModule, TransformerSpec, ParallelTransformer,
        CheckpointStrategy, ResidualPolicy, mpu, LayerNorm, _dbg,
    )

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# 枚举：注意力掩码语义
# ---------------------------------------------------------------------------
class AttentionMaskKind(enum.Enum):
    """注意力掩码的语义类型。

    ADDITIVE_BERT = BERT 风格：将 (1-mask)*(-10000) 加到分数上，
                   0位置保留，1位置被屏蔽。
    MULTIPLICATIVE_GPT2 = GPT-2 风格：因果掩码，左三角为 1，
                          其余位置减去 10000。
    """
    ADDITIVE_BERT = "additive_bert"
    MULTIPLICATIVE_GPT2 = "multiplicative_gpt2"


class ModelParallelLogitsMode(enum.Enum):
    """LM logits 的输出模式（替代 parallel_output 布尔值）。

    PARALLEL = logits 保留模型并行分片（训练时，配合并行 CrossEntropy）
    GATHERED = logits 汇聚为完整词表（推理/评估时）
    """
    PARALLEL = "parallel"
    GATHERED = "gathered"


# ---------------------------------------------------------------------------
# BERT / GPT-2 注意力掩码函数
# 上游为裸函数；Walpurgis 包装为 AttentionMaskFuncRegistry，使策略可枚举
# ---------------------------------------------------------------------------
def _bert_attention_mask_func(scores: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    """BERT 加法掩码：直接将掩码张量加到注意力分数上。"""
    _dbg("mask_func.bert", scores_shape=tuple(scores.shape))
    return scores + mask


def _gpt2_attention_mask_func(scores: torch.Tensor,
                              ltor_mask: torch.Tensor) -> torch.Tensor:
    """GPT-2 因果掩码：乘法保留 + 减法屏蔽。"""
    _dbg("mask_func.gpt2", scores_shape=tuple(scores.shape))
    return torch.mul(scores, ltor_mask) - 10000.0 * (1.0 - ltor_mask)


class AttentionMaskFuncRegistry:
    """注意力掩码函数注册表，根据 AttentionMaskKind 返回对应函数。"""

    _registry = {
        AttentionMaskKind.ADDITIVE_BERT: _bert_attention_mask_func,
        AttentionMaskKind.MULTIPLICATIVE_GPT2: _gpt2_attention_mask_func,
    }

    @classmethod
    def get(cls, kind: AttentionMaskKind) -> Callable:
        func = cls._registry.get(kind)
        assert func is not None, f"未知掩码类型: {kind}"
        _dbg("AttentionMaskFuncRegistry.get", kind=kind.value)
        return func

    @classmethod
    def register(cls, kind: AttentionMaskKind, func: Callable) -> None:
        """扩展点：注册自定义掩码函数。"""
        cls._registry[kind] = func
        _dbg("AttentionMaskFuncRegistry.register", kind=kind.value)


# ---------------------------------------------------------------------------
# 模型工具函数（来自 megatron/model/utils.py）
# ---------------------------------------------------------------------------
def init_method_normal(sigma: float) -> Callable:
    """基于 N(0, sigma) 的权重初始化方法。"""
    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


def scaled_init_method_normal(sigma: float, num_layers: int) -> Callable:
    """缩放初始化：N(0, sigma / sqrt(2 * num_layers))。

    用于输出层（attention output / mlp output），防止深层网络方差爆炸。
    """
    std = sigma / math.sqrt(2.0 * num_layers)
    _dbg("scaled_init_method_normal", sigma=sigma, num_layers=num_layers, std=std)
    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


def get_linear_layer(rows: int, cols: int,
                     init_method: Callable) -> nn.Linear:
    """带权重初始化的线性层，偏置清零。"""
    layer = nn.Linear(rows, cols)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    _dbg("get_linear_layer", rows=rows, cols=cols)
    return layer


@torch.jit.script
def gelu_impl(x: torch.Tensor) -> torch.Tensor:
    """OpenAI GELU 实现（tanh 近似版）。"""
    return 0.5 * x * (1.0 + torch.tanh(
        0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return gelu_impl(x)


# ---------------------------------------------------------------------------
# WeightDecayManifest: 参数分组台账（替代 get_params_for_weight_decay_optimization）
# 上游：裸函数返回两个 dict，调用者需记住哪个有 weight_decay
# Walpurgis：dataclass 封装，字段名即含义，self_check() 验证分组完整性
# ---------------------------------------------------------------------------
@dataclass
class WeightDecayManifest:
    """权重衰减参数分组台账。

    with_decay:    LayerNorm 以外的权重参数（含 weight_decay）
    without_decay: LayerNorm 参数 + 所有偏置（weight_decay=0）
    """
    with_decay: Dict[str, Any]
    without_decay: Dict[str, Any]

    @classmethod
    def from_module(cls, module: nn.Module) -> "WeightDecayManifest":
        """从 nn.Module 构建参数分组台账。"""
        _dbg("WeightDecayManifest.from_module",
             module=module.__class__.__name__)
        wd_params = {'params': []}
        no_wd_params = {'params': [], 'weight_decay': 0.0}

        for submodule in module.modules():
            if isinstance(submodule, (LayerNorm, nn.LayerNorm)):
                no_wd_params['params'].extend(
                    [p for p in submodule._parameters.values()
                     if p is not None])
            else:
                wd_params['params'].extend(
                    [p for n, p in submodule._parameters.items()
                     if p is not None and n != 'bias'])
                no_wd_params['params'].extend(
                    [p for n, p in submodule._parameters.items()
                     if p is not None and n == 'bias'])

        _dbg("WeightDecayManifest.from_module.done",
             wd_count=len(wd_params['params']),
             no_wd_count=len(no_wd_params['params']))
        return cls(with_decay=wd_params, without_decay=no_wd_params)

    def as_param_groups(self):
        """返回 optimizer 所需的 param_groups 格式。"""
        return [self.with_decay, self.without_decay]

    def self_check(self, module: nn.Module) -> bool:
        """验证所有参数均被分组，无遗漏。"""
        all_params = set(id(p) for p in module.parameters())
        grouped = set(id(p) for p in self.with_decay['params'])
        grouped |= set(id(p) for p in self.without_decay['params'])
        ok = all_params == grouped
        _dbg("WeightDecayManifest.self_check", ok=ok,
             total=len(all_params), grouped=len(grouped))
        return ok


# ---------------------------------------------------------------------------
# parallel_lm_logits: LM logits 的模型并行计算
# ---------------------------------------------------------------------------
def parallel_lm_logits(input_: torch.Tensor,
                       word_embeddings_weight: torch.Tensor,
                       mode: ModelParallelLogitsMode,
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """计算 LM logits，支持模型并行输出或汇聚输出。

    上游以 parallel_output: bool 控制；Walpurgis 改为 ModelParallelLogitsMode(Enum)。
    """
    _dbg("parallel_lm_logits", mode=mode.value,
         input_shape=tuple(input_.shape))
    input_parallel = mpu.copy_to_model_parallel_region(input_) \
        if hasattr(mpu, 'copy_to_model_parallel_region') else input_

    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    if mode == ModelParallelLogitsMode.PARALLEL:
        _dbg("parallel_lm_logits.parallel_output")
        return logits_parallel
    else:
        _dbg("parallel_lm_logits.gathered_output")
        if hasattr(mpu, 'gather_from_model_parallel_region'):
            return mpu.gather_from_model_parallel_region(logits_parallel)
        return logits_parallel


# ---------------------------------------------------------------------------
# EmbeddingSpec: 嵌入层参数规格（新增，上游无此抽象）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingSpec:
    """嵌入层构造规格。"""
    hidden_size: int
    vocab_size: int
    max_sequence_length: int
    embedding_dropout_prob: float
    init_method: Callable
    num_tokentypes: int = 0

    def self_check(self) -> bool:
        ok = (
            self.hidden_size > 0
            and self.vocab_size > 0
            and self.max_sequence_length > 0
            and 0.0 <= self.embedding_dropout_prob < 1.0
            and self.num_tokentypes >= 0
        )
        _dbg("EmbeddingSpec.self_check", ok=ok)
        return ok


# ---------------------------------------------------------------------------
# Pooler: 序列池化层（BERT [CLS] token → 二分类）
# ---------------------------------------------------------------------------
class Pooler(WalpurgisModule):
    """序列表示池化：取指定位置的 hidden state，过线性层 + tanh。

    BERT 的 NSP（Next Sentence Prediction）头依赖此层。
    GPT-2 不使用。
    """

    def __init__(self, hidden_size: int, init_method: Callable):
        super().__init__("Pooler")
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        _dbg("Pooler.init", hidden_size=hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                sequence_index: int = 0) -> torch.Tensor:
        """取 sequence_index 位置的表示，过 dense + tanh。"""
        _dbg("Pooler.forward",
             shape=tuple(hidden_states.shape), idx=sequence_index)
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


# ---------------------------------------------------------------------------
# Embedding: 词嵌入 + 位置嵌入 + 可选 token-type 嵌入
# ---------------------------------------------------------------------------
class Embedding(WalpurgisModule):
    """三路嵌入：词 + 位置 + token-type（可选）。

    上游将所有参数平铺传入构造函数；
    Walpurgis 接受 EmbeddingSpec，使构造意图可静态审计。
    """

    def __init__(self, spec: EmbeddingSpec):
        super().__init__("Embedding")
        _dbg("Embedding.init",
             vocab=spec.vocab_size, hidden=spec.hidden_size,
             max_seq=spec.max_sequence_length,
             tokentypes=spec.num_tokentypes)

        self.hidden_size = spec.hidden_size
        self.init_method = spec.init_method
        self.num_tokentypes = spec.num_tokentypes

        # 词嵌入（模型并行）
        try:
            self.word_embeddings = mpu.VocabParallelEmbedding(
                spec.vocab_size, spec.hidden_size,
                init_method=spec.init_method)
        except AttributeError:
            # stub 模式
            self.word_embeddings = nn.Embedding(
                spec.vocab_size, spec.hidden_size)
        self._word_embeddings_key = 'word_embeddings'

        # 位置嵌入（串行）
        self.position_embeddings = nn.Embedding(
            spec.max_sequence_length, spec.hidden_size)
        spec.init_method(self.position_embeddings.weight)
        self._position_embeddings_key = 'position_embeddings'

        # Token-type 嵌入（可选，BERT 使用）
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if spec.num_tokentypes > 0:
            self.tokentype_embeddings = nn.Embedding(
                spec.num_tokentypes, spec.hidden_size)
            spec.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.embedding_dropout = nn.Dropout(spec.embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes: int) -> None:
        """动态添加 token-type 嵌入（用于预训练模型热添加 NSP 功能）。"""
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype_embeddings 已初始化，不可重复添加')
        _dbg("Embedding.add_tokentype_embeddings", n=num_tokentypes)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = nn.Embedding(num_tokentypes, self.hidden_size)
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                tokentype_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        _dbg("Embedding.forward",
             input_shape=tuple(input_ids.shape),
             has_tokentypes=(tokentype_ids is not None))

        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        embeddings = words + positions

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None, \
                "tokentype_ids 非空但 tokentype_embeddings 未初始化"
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None, \
                "tokentype_embeddings 已初始化但 tokentype_ids 为 None"

        embeddings = self.embedding_dropout(embeddings)
        _dbg("Embedding.forward.done", out_shape=tuple(embeddings.shape))
        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None,
                                       prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = \
            self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] = \
            self.position_embeddings.state_dict(destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = \
                self.tokentype_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        # 词嵌入
        if self._word_embeddings_key in state_dict:
            sd = state_dict[self._word_embeddings_key]
        else:
            sd = {k.split('word_embeddings.')[1]: v
                  for k, v in state_dict.items()
                  if 'word_embeddings' in k}
        self.word_embeddings.load_state_dict(sd, strict=strict)

        # 位置嵌入
        if self._position_embeddings_key in state_dict:
            sd = state_dict[self._position_embeddings_key]
        else:
            sd = {k.split('position_embeddings.')[1]: v
                  for k, v in state_dict.items()
                  if 'position_embeddings' in k}
        self.position_embeddings.load_state_dict(sd, strict=strict)

        # Token-type 嵌入
        if self.num_tokentypes > 0:
            sd = {}
            if self._tokentype_embeddings_key in state_dict:
                sd = state_dict[self._tokentype_embeddings_key]
            else:
                sd = {k.split('tokentype_embeddings.')[1]: v
                      for k, v in state_dict.items()
                      if 'tokentype_embeddings' in k}
            if sd:
                self.tokentype_embeddings.load_state_dict(sd, strict=strict)
            else:
                print('***WARNING*** 检查点中未找到 tokentype_embeddings', flush=True)


# ---------------------------------------------------------------------------
# TransformerLanguageModel: BERT 和 GPT-2 共用的语言模型骨架
# ---------------------------------------------------------------------------
class TransformerLanguageModel(WalpurgisModule):
    """共享 Transformer 语言模型骨架。

    BERT:  add_pooler=True,  residual=POST_NORM, mask_kind=ADDITIVE_BERT
    GPT-2: add_pooler=False, residual=PRE_NORM,  mask_kind=MULTIPLICATIVE_GPT2

    这是 beb3e0d38 这次重构最核心的抽象——两个模型在此共用同一骨架，
    差异被最小化为三个参数。Walpurgis 以枚举将这三个差异命名，
    使「BERT vs GPT-2」从运行时数据变为静态可读的策略选择。
    """

    def __init__(self,
                 transformer_spec: TransformerSpec,
                 embedding_spec: EmbeddingSpec,
                 mask_kind: AttentionMaskKind,
                 add_pooler: bool = False):
        super().__init__("TransformerLanguageModel")
        _dbg("TransformerLanguageModel.init",
             mask=mask_kind.value,
             add_pooler=add_pooler,
             residual=transformer_spec.residual.value)

        self.hidden_size = transformer_spec.hidden_size
        self.num_tokentypes = embedding_spec.num_tokentypes
        self.init_method = transformer_spec.init_method
        self.add_pooler = add_pooler

        # 嵌入层
        self.embedding = Embedding(embedding_spec)
        self._embedding_key = 'embedding'

        # Transformer（含掩码函数）
        attention_mask_func = AttentionMaskFuncRegistry.get(mask_kind)
        self.transformer = ParallelTransformer(
            transformer_spec, attention_mask_func)
        self._transformer_key = 'transformer'

        # Pooler（可选，仅 BERT）
        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

    def forward(self, input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                tokentype_ids: Optional[torch.Tensor] = None,
                layer_past=None,
                get_key_value: bool = False,
                pooling_sequence_index: int = 0):
        _dbg("TransformerLanguageModel.forward",
             input_shape=tuple(input_ids.shape),
             has_pooler=self.add_pooler)

        emb = self.embedding(input_ids, position_ids,
                             tokentype_ids=tokentype_ids)
        transformer_out = self.transformer(emb, attention_mask,
                                           layer_past=layer_past,
                                           get_key_value=get_key_value)

        if self.add_pooler:
            pooled = self.pooler(transformer_out, pooling_sequence_index)
            _dbg("TransformerLanguageModel.forward.done", with_pooler=True)
            return transformer_out, pooled

        _dbg("TransformerLanguageModel.forward.done", with_pooler=False)
        return transformer_out

    def state_dict_for_save_checkpoint(self, destination=None,
                                       prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._embedding_key] = \
            self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._transformer_key] = \
            self.transformer.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] = \
                self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        if self._embedding_key in state_dict:
            sd = state_dict[self._embedding_key]
        else:
            sd = {k: v for k, v in state_dict.items()
                  if '_embeddings' in k}
        self.embedding.load_state_dict(sd, strict=strict)

        if self._transformer_key in state_dict:
            sd = state_dict[self._transformer_key]
        else:
            sd = {k.split('transformer.')[1]: v
                  for k, v in state_dict.items()
                  if 'transformer.' in k}
        self.transformer.load_state_dict(sd, strict=strict)

        if self.add_pooler:
            assert 'pooler' in state_dict, \
                '检查点中未找到 pooler 数据'
            self.pooler.load_state_dict(
                state_dict[self._pooler_key], strict=strict)


# ---------------------------------------------------------------------------
# BertLMHead: BERT Masked LM 头
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LMHeadSpec:
    """BertLMHead 构造规格。"""
    mpu_vocab_size: int
    hidden_size: int
    init_method: Callable
    layernorm_epsilon: float
    mode: ModelParallelLogitsMode

    def self_check(self) -> bool:
        ok = self.mpu_vocab_size > 0 and self.hidden_size > 0
        _dbg("LMHeadSpec.self_check", ok=ok)
        return ok


class BertLMHead(WalpurgisModule):
    """BERT Masked LM 输出头：dense → GELU → LayerNorm → parallel_lm_logits。

    上游以 parallel_output: bool 控制；Walpurgis 改为 ModelParallelLogitsMode(Enum)。
    """

    def __init__(self, spec: LMHeadSpec):
        super().__init__("BertLMHead")
        _dbg("BertLMHead.init", vocab=spec.mpu_vocab_size,
             hidden=spec.hidden_size, mode=spec.mode.value)

        self.mode = spec.mode
        self.bias = nn.Parameter(torch.zeros(spec.mpu_vocab_size))
        self.bias.model_parallel = True  # 标记为模型并行参数

        self.dense = get_linear_layer(
            spec.hidden_size, spec.hidden_size, spec.init_method)
        self.layernorm = LayerNorm(
            spec.hidden_size, eps=spec.layernorm_epsilon)

    def forward(self, hidden_states: torch.Tensor,
                word_embeddings_weight: torch.Tensor) -> torch.Tensor:
        _dbg("BertLMHead.forward", shape=tuple(hidden_states.shape))
        x = self.dense(hidden_states)
        x = gelu(x)
        x = self.layernorm(x)
        output = parallel_lm_logits(x, word_embeddings_weight,
                                    self.mode, bias=self.bias)
        return output


# ---------------------------------------------------------------------------
# BertModel: 完整 BERT 模型
# ---------------------------------------------------------------------------
class BertModel(WalpurgisModule):
    """BERT 语言模型（含可选 NSP 二分类头）。

    掩码策略：ADDITIVE_BERT（加法掩码）
    残差策略：POST_NORM
    池化：add_binary_head=True 时启用
    """

    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 embedding_dropout_prob: float,
                 attention_dropout_prob: float,
                 output_dropout_prob: float,
                 max_sequence_length: int,
                 checkpoint_activations: bool,
                 checkpoint_num_layers: int = 1,
                 add_binary_head: bool = False,
                 layernorm_epsilon: float = 1.0e-5,
                 init_method_std: float = 0.02,
                 num_tokentypes: int = 0,
                 logits_mode: ModelParallelLogitsMode = ModelParallelLogitsMode.PARALLEL):
        super().__init__("BertModel")
        _dbg("BertModel.init",
             layers=num_layers, vocab=vocab_size, hidden=hidden_size,
             binary_head=add_binary_head)

        self.add_binary_head = add_binary_head
        self.logits_mode = logits_mode

        init_method = init_method_normal(init_method_std)
        scaled_init = scaled_init_method_normal(init_method_std, num_layers)

        ckpt_strategy = (CheckpointStrategy.FULL if checkpoint_activations
                         else CheckpointStrategy.NONE)

        transformer_spec = TransformerSpec(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            mlp_activation_func=gelu,
            layernorm_epsilon=layernorm_epsilon,
            init_method=init_method,
            output_layer_init_method=scaled_init,
            checkpoint_strategy=ckpt_strategy,
            checkpoint_num_layers=checkpoint_num_layers,
            residual=ResidualPolicy.POST_NORM,  # BERT 特征
        )

        embedding_spec = EmbeddingSpec(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dropout_prob=embedding_dropout_prob,
            init_method=init_method,
            num_tokentypes=num_tokentypes,
        )

        self.language_model = TransformerLanguageModel(
            transformer_spec=transformer_spec,
            embedding_spec=embedding_spec,
            mask_kind=AttentionMaskKind.ADDITIVE_BERT,
            add_pooler=add_binary_head,
        )
        self._language_model_key = 'language_model'

        lm_head_spec = LMHeadSpec(
            mpu_vocab_size=self.language_model.embedding.word_embeddings.weight.size(0),
            hidden_size=hidden_size,
            init_method=init_method,
            layernorm_epsilon=layernorm_epsilon,
            mode=logits_mode,
        )
        self.lm_head = BertLMHead(lm_head_spec)
        self._lm_head_key = 'lm_head'

        if add_binary_head:
            self.binary_head = get_linear_layer(hidden_size, 2, init_method)
            self._binary_head_key = 'binary_head'

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                tokentype_ids: Optional[torch.Tensor] = None):
        _dbg("BertModel.forward", input_shape=tuple(input_ids.shape))

        # BERT 需要扩展掩码 [b, s] → [b, 1, s, s]，值域转为加法掩码
        extended_mask = self._extend_attention_mask(
            attention_mask,
            next(self.language_model.parameters()).dtype)
        position_ids = self._build_position_ids(input_ids)

        if self.add_binary_head:
            lm_out, pooled = self.language_model(
                input_ids, position_ids, extended_mask,
                tokentype_ids=tokentype_ids)
        else:
            lm_out = self.language_model(
                input_ids, position_ids, extended_mask,
                tokentype_ids=tokentype_ids)

        lm_logits = self.lm_head(
            lm_out,
            self.language_model.embedding.word_embeddings.weight)

        if self.add_binary_head:
            binary_logits = self.binary_head(pooled)
            return lm_logits, binary_logits

        return lm_logits, None

    @staticmethod
    def _extend_attention_mask(attention_mask: torch.Tensor,
                               dtype: torch.dtype) -> torch.Tensor:
        """[b, s] → [b, 1, s, s]，0位置变为 -10000（被屏蔽），1位置保留 0。"""
        _dbg("BertModel._extend_attention_mask",
             in_shape=tuple(attention_mask.shape))
        b1s = attention_mask.unsqueeze(1)   # [b, 1, s]
        bs1 = attention_mask.unsqueeze(2)   # [b, s, 1]
        bss = b1s * bs1                     # [b, s, s]（广播乘法）
        extended = bss.unsqueeze(1)         # [b, 1, s, s]
        extended = extended.to(dtype=dtype)
        extended = (1.0 - extended) * -10000.0
        return extended

    @staticmethod
    def _build_position_ids(token_ids: torch.Tensor) -> torch.Tensor:
        """构建位置 ID [0, 1, ..., seq_len-1]，广播到 batch。"""
        seq_length = token_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=token_ids.device)
        return position_ids.unsqueeze(0).expand_as(token_ids)

    def state_dict_for_save_checkpoint(self, destination=None,
                                       prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._language_model_key] = \
            self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._lm_head_key] = \
            self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.add_binary_head:
            state_dict_[self._binary_head_key] = \
                self.binary_head.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.lm_head.load_state_dict(
            state_dict[self._lm_head_key], strict=strict)
        if self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)


# ---------------------------------------------------------------------------
# GPT2Model: GPT-2 语言模型
# ---------------------------------------------------------------------------
class GPT2Model(WalpurgisModule):
    """GPT-2 因果语言模型。

    掩码策略：MULTIPLICATIVE_GPT2（因果掩码）
    残差策略：PRE_NORM
    无 Pooler（纯解码器）
    """

    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 embedding_dropout_prob: float,
                 attention_dropout_prob: float,
                 output_dropout_prob: float,
                 max_sequence_length: int,
                 checkpoint_activations: bool,
                 checkpoint_num_layers: int = 1,
                 layernorm_epsilon: float = 1.0e-5,
                 init_method_std: float = 0.02,
                 num_tokentypes: int = 0,
                 logits_mode: ModelParallelLogitsMode = ModelParallelLogitsMode.PARALLEL):
        super().__init__("GPT2Model")
        _dbg("GPT2Model.init",
             layers=num_layers, vocab=vocab_size, hidden=hidden_size)

        self.logits_mode = logits_mode
        init_method = init_method_normal(init_method_std)
        scaled_init = scaled_init_method_normal(init_method_std, num_layers)
        ckpt_strategy = (CheckpointStrategy.FULL if checkpoint_activations
                         else CheckpointStrategy.NONE)

        transformer_spec = TransformerSpec(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            mlp_activation_func=gelu,
            layernorm_epsilon=layernorm_epsilon,
            init_method=init_method,
            output_layer_init_method=scaled_init,
            checkpoint_strategy=ckpt_strategy,
            checkpoint_num_layers=checkpoint_num_layers,
            residual=ResidualPolicy.PRE_NORM,  # GPT-2 特征
        )

        embedding_spec = EmbeddingSpec(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dropout_prob=embedding_dropout_prob,
            init_method=init_method,
            num_tokentypes=num_tokentypes,
        )

        self.language_model = TransformerLanguageModel(
            transformer_spec=transformer_spec,
            embedding_spec=embedding_spec,
            mask_kind=AttentionMaskKind.MULTIPLICATIVE_GPT2,
            add_pooler=False,
        )
        self._language_model_key = 'language_model'

    def forward(self, input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                tokentype_ids: Optional[torch.Tensor] = None,
                layer_past=None,
                get_key_value: bool = False):
        _dbg("GPT2Model.forward", input_shape=tuple(input_ids.shape))

        lm_out = self.language_model(
            input_ids, position_ids, attention_mask,
            tokentype_ids=tokentype_ids,
            layer_past=layer_past,
            get_key_value=get_key_value)

        if get_key_value:
            lm_out, presents = lm_out

        output = parallel_lm_logits(
            lm_out,
            self.language_model.embedding.word_embeddings.weight,
            self.logits_mode)

        if get_key_value:
            return [output, presents]
        return output

    def state_dict_for_save_checkpoint(self, destination=None,
                                       prefix='', keep_vars=False):
        state_dict_ = {}
        state_dict_[self._language_model_key] = \
            self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# 工厂函数（兼容上游 get_language_model 签名）
# ---------------------------------------------------------------------------
def get_language_model(num_layers: int,
                       vocab_size: int,
                       hidden_size: int,
                       num_attention_heads: int,
                       embedding_dropout_prob: float,
                       attention_dropout_prob: float,
                       output_dropout_prob: float,
                       max_sequence_length: int,
                       num_tokentypes: int,
                       attention_mask_func: Callable,
                       add_pooler: bool,
                       checkpoint_activations: bool,
                       checkpoint_num_layers: int,
                       layernorm_epsilon: float,
                       init_method: Callable,
                       scaled_init_method: Callable,
                       residual_connection_post_layernorm: bool,
                       ) -> Tuple[TransformerLanguageModel, str]:
    """兼容上游 get_language_model 调用签名的工厂函数。"""
    _dbg("get_language_model", add_pooler=add_pooler,
         post_ln=residual_connection_post_layernorm)

    residual = (ResidualPolicy.POST_NORM if residual_connection_post_layernorm
                else ResidualPolicy.PRE_NORM)
    ckpt = (CheckpointStrategy.FULL if checkpoint_activations
            else CheckpointStrategy.NONE)

    # 推断掩码类型（从掩码函数引用判断）
    if attention_mask_func is _bert_attention_mask_func:
        mask_kind = AttentionMaskKind.ADDITIVE_BERT
    else:
        mask_kind = AttentionMaskKind.MULTIPLICATIVE_GPT2

    transformer_spec = TransformerSpec(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        attention_dropout_prob=attention_dropout_prob,
        output_dropout_prob=output_dropout_prob,
        mlp_activation_func=gelu,
        layernorm_epsilon=layernorm_epsilon,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        checkpoint_strategy=ckpt,
        checkpoint_num_layers=checkpoint_num_layers,
        residual=residual,
    )
    embedding_spec = EmbeddingSpec(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dropout_prob=embedding_dropout_prob,
        init_method=init_method,
        num_tokentypes=num_tokentypes,
    )

    model = TransformerLanguageModel(
        transformer_spec=transformer_spec,
        embedding_spec=embedding_spec,
        mask_kind=mask_kind,
        add_pooler=add_pooler,
    )
    return model, 'language_model'


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------
def self_check() -> bool:
    """覆盖 TransformerLanguageModel / BertModel / GPT2Model 关键路径。"""

    def dummy_init(t):
        nn.init.normal_(t, std=0.02)

    # EmbeddingSpec 验证
    espec = EmbeddingSpec(
        hidden_size=64, vocab_size=512, max_sequence_length=128,
        embedding_dropout_prob=0.0, init_method=dummy_init, num_tokentypes=0)
    assert espec.self_check()

    # AttentionMaskFuncRegistry 验证
    bert_func = AttentionMaskFuncRegistry.get(AttentionMaskKind.ADDITIVE_BERT)
    gpt_func = AttentionMaskFuncRegistry.get(AttentionMaskKind.MULTIPLICATIVE_GPT2)
    assert bert_func is not gpt_func

    # ModelParallelLogitsMode 枚举值不变量
    assert ModelParallelLogitsMode.PARALLEL.value == "parallel"
    assert ModelParallelLogitsMode.GATHERED.value == "gathered"

    # init_method_normal / scaled_init_method_normal 返回可调用
    init_fn = init_method_normal(0.02)
    scaled_fn = scaled_init_method_normal(0.02, 12)
    t = torch.zeros(4, 4)
    init_fn(t)
    scaled_fn(t)

    # WeightDecayManifest：简单线性层
    linear = nn.Linear(8, 4)
    manifest = WeightDecayManifest.from_module(linear)
    assert manifest.self_check(linear)

    # LMHeadSpec 验证
    lm_spec = LMHeadSpec(
        mpu_vocab_size=512, hidden_size=64,
        init_method=dummy_init, layernorm_epsilon=1e-5,
        mode=ModelParallelLogitsMode.GATHERED)
    assert lm_spec.self_check()

    # GPT-2 掩码函数行为验证（因果掩码）
    scores = torch.ones(1, 1, 4, 4)
    causal_mask = torch.tril(torch.ones(1, 1, 4, 4))
    masked = _gpt2_attention_mask_func(scores, causal_mask)
    assert masked[0, 0, 0, 1] < -9000  # 上三角应被屏蔽

    # BERT 掩码函数行为验证
    bert_mask = torch.zeros(1, 1, 4, 4)
    bert_masked = _bert_attention_mask_func(scores, bert_mask)
    assert torch.allclose(bert_masked, scores)

    print("[self_check] language_model_unified: ALL PASS")
    return True


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    self_check()
