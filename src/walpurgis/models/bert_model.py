"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/model/bert_model.py（新增文件，218行）

鲁迅拿法改写（≥20%）：
  上游将原来散布在 pretrain_bert.py 里的模型定义（528行 → 抽出来，
  剩 pretrain_bert.py 大幅瘦身）集中到 bert_model.py。
  BERT 的结构本来就清晰：token_cls_head + nsp_head。
  但上游代码把两个任务头的权重初始化、loss 计算都塞进同一个 forward，
  边前向、边算 loss，边返回，一锅烩。
  鲁迅说：「什么都放在一起，就什么都看不清楚。」

  Walpurgis 改写要点：
  1. MaskedLMHead: 上游 BertLMHead（dense→gelu→layernorm→weight_tying projection），
     Walpurgis 命名更直白，且显式标注 weight tying 语义。
  2. NextSentencePredHead: 上游 BertPooler + 二分类 Linear 的组合，
     Walpurgis 将两者合并为单一预测头，语义清晰。
  3. BertModelWrapper: 替代上游 BertModel，持有 LanguageModelCore + 两个任务头，
     forward 返回 (mlm_logits, nsp_logits) 的 BertOutput dataclass，
     与 loss 计算完全解耦（上游混在一起）。
  4. BertLossComputer: 将 MLM loss + NSP loss 从 pretrain_bert.py 抽出，
     作为独立单元（上游散落在 pretrain 脚本）。

迁移位置: src/walpurgis/models/bert_model.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：bert_model 前向关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:bert_model:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:bert_model:{tag}] {msg}", file=sys.stderr, flush=True)


# ── MaskedLMHead ──────────────────────────────────────────────────────────────
# 上游 BertLMHead：dense(hidden→hidden) + gelu + LayerNorm + 到词表的 weight-tied 投影。
# weight tying：projection.weight = word_embeddings.weight（上游代码里悄悄 set 的）。

class MaskedLMHead(nn.Module):
    """
    Masked Language Modeling 预测头（对应上游 BertLMHead）。

    上游把 weight tying 写在 BertModel.__init__ 里：
        self.lm_head.decoder.weight = self.word_embeddings.weight
    Walpurgis 将 weight tying 作为显式方法 `tie_weights(word_embeddings)`，
    调用者主动调用，不再隐式依赖初始化顺序。

    前向路径：
        hidden [B,T,H] → dense → GELU → LayerNorm → decoder → logits [B,T,V]
    """

    def __init__(self, hidden_size: int, vocab_size: int, init_method) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        init_method(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

        self.layernorm = nn.LayerNorm(hidden_size)

        # decoder：hidden_size → vocab_size，weight 将被 tie 到 word_embeddings
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        init_method(self.decoder.weight)
        _dbg("MLM_HEAD_INIT", f"hidden={hidden_size}, vocab={vocab_size}")

    def tie_weights(self, word_embedding_weight: torch.Tensor) -> None:
        """
        将 decoder.weight 与 word_embeddings.weight 绑定。
        上游：隐式在 BertModel.__init__ 末尾执行。
        Walpurgis：显式调用，语义清晰。
        """
        assert self.decoder.weight.shape == word_embedding_weight.shape, (
            f"Weight tying shape mismatch: "
            f"decoder={self.decoder.weight.shape}, "
            f"word_emb={word_embedding_weight.shape}"
        )
        self.decoder.weight = word_embedding_weight
        _dbg("MLM_WEIGHT_TIED", f"shape={list(word_embedding_weight.shape)}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        returns:       [B, T, V]   (MLM logits)
        """
        _dbg("MLM_HEAD_INPUT", hidden_states)
        x = F.gelu(self.dense(hidden_states))
        _dbg("MLM_HEAD_GELU", x)
        x = self.layernorm(x)
        _dbg("MLM_HEAD_NORMED", x)
        logits = self.decoder(x)
        _dbg("MLM_HEAD_LOGITS", logits)
        return logits


# ── NextSentencePredHead ──────────────────────────────────────────────────────
# 上游：BertModel 里有 PoolerHead（见 language_model.py）+ 单独的 binary Linear。
# Walpurgis：合并为 NextSentencePredHead，从 [CLS] pooled 输出直接到 2-class logits。

class NextSentencePredHead(nn.Module):
    """
    Next Sentence Prediction 预测头（对应上游 nsp_head）。

    输入：PoolerHead 的输出 [B, H]（已经过 Tanh 激活）
    输出：[B, 2]  (IsNext / NotNext logits)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        _dbg("NSP_HEAD_INIT", f"hidden={hidden_size}")

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        pooled_output: [B, H]  (from PoolerHead)
        returns:       [B, 2]  (NSP logits)
        """
        _dbg("NSP_HEAD_INPUT", pooled_output)
        logits = self.classifier(pooled_output)
        _dbg("NSP_HEAD_LOGITS", logits)
        return logits


# ── BertOutput dataclass ──────────────────────────────────────────────────────
# 上游 forward 直接返回 loss，不分离 logits。
# Walpurgis：结构化输出，loss 计算交给 BertLossComputer。

@dataclass
class BertOutput:
    """BertModelWrapper.forward 的结构化返回值（上游是裸 loss，Walpurgis 解耦）"""
    mlm_logits: torch.Tensor        # [B, T, V]
    nsp_logits: torch.Tensor        # [B, 2]
    hidden_states: torch.Tensor     # [B, T, H]  最终 transformer 输出
    pooled_output: torch.Tensor     # [B, H]     [CLS] 池化结果


# ── BertModelWrapper ──────────────────────────────────────────────────────────
# 替代上游 BertModel，持有 LanguageModelCore + MaskedLMHead + NextSentencePredHead。

class BertModelWrapper(nn.Module):
    """
    BERT 全模型（Walpurgis 版）。

    对应上游 BertModel（megatron/model/bert_model.py 中 class BertModel）。
    上游 forward() 接收 labels 并计算 loss，Walpurgis 只返回 logits（BertOutput）。
    loss 计算由 BertLossComputer 负责，遵循关注点分离。

    依赖：
        language_model: LanguageModelCore(add_pooler=True)
        vocab_size: 与 word_embeddings 一致，用于构造 MaskedLMHead
    """

    def __init__(
        self,
        language_model,          # LanguageModelCore 实例（add_pooler=True）
        hidden_size: int,
        vocab_size: int,
        init_method,
    ) -> None:
        super().__init__()
        self.language_model = language_model

        self.mlm_head = MaskedLMHead(hidden_size, vocab_size, init_method)
        self.nsp_head = NextSentencePredHead(hidden_size)

        # weight tying（上游在构造时完成，Walpurgis 显式调用）
        word_emb_weight = language_model.embedding.word_embeddings.weight
        self.mlm_head.tie_weights(word_emb_weight)

        _dbg("BERT_WRAPPER_INIT",
             f"hidden={hidden_size}, vocab={vocab_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> BertOutput:
        """
        input_ids:      [B, T]
        attention_mask: [B, T] or [B, 1, T, T]
        tokentype_ids:  [B, T] or None  (BERT segment A/B)
        position_ids:   [B, T] or None  (若 None，自动生成 0..T-1)

        返回：BertOutput (mlm_logits, nsp_logits, hidden_states, pooled_output)
        """
        B, T = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            _dbg("FWD_POS_AUTO", position_ids)

        _dbg("FWD_INPUT_IDS", input_ids)
        _dbg("FWD_ATTN_MASK", attention_mask)

        # LanguageModelCore with add_pooler=True → (hidden_states, pooled_output)
        hidden_states, pooled_output = self.language_model(
            input_ids, position_ids, attention_mask, tokentype_ids
        )
        _dbg("FWD_HIDDEN", hidden_states)
        _dbg("FWD_POOLED", pooled_output)

        mlm_logits = self.mlm_head(hidden_states)
        _dbg("FWD_MLM_LOGITS", mlm_logits)

        nsp_logits = self.nsp_head(pooled_output)
        _dbg("FWD_NSP_LOGITS", nsp_logits)

        return BertOutput(
            mlm_logits=mlm_logits,
            nsp_logits=nsp_logits,
            hidden_states=hidden_states,
            pooled_output=pooled_output,
        )


# ── BertLossComputer ──────────────────────────────────────────────────────────
# 上游在 pretrain_bert.py 的 forward_step() / loss_func() 里散落着 MLM+NSP loss。
# Walpurgis 将其集中为独立单元（上游的 CrossEntropyLoss + Binary NSP loss）。

@dataclass
class BertLossOutput:
    """BertLossComputer 的结构化返回"""
    total_loss: torch.Tensor    # MLM loss + NSP loss
    mlm_loss: torch.Tensor      # 仅 MLM 部分
    nsp_loss: torch.Tensor      # 仅 NSP 部分
    mlm_num_tokens: int         # 参与 MLM 损失计算的 token 数


class BertLossComputer:
    """
    BERT 双任务损失计算（上游散落在 pretrain_bert.py，Walpurgis 集中管理）。

    MLM loss：CrossEntropy，仅在 lm_labels != -1 的位置计算（-1 为 ignore_index）。
    NSP loss：CrossEntropy，二分类（IsNext=0, NotNext=1）。

    上游代码：
        lm_loss = mpu.vocab_parallel_cross_entropy(output, lm_labels)
        nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    Walpurgis：语义等价，去掉 mpu 并行依赖，用标准 F.cross_entropy。
    """

    def __init__(self, lm_loss_weight: float = 1.0, nsp_loss_weight: float = 1.0) -> None:
        self.lm_loss_weight = lm_loss_weight
        self.nsp_loss_weight = nsp_loss_weight
        _dbg("LOSS_INIT",
             f"lm_weight={lm_loss_weight}, nsp_weight={nsp_loss_weight}")

    def __call__(
        self,
        bert_output: BertOutput,
        lm_labels: torch.Tensor,      # [B, T]  masked token ids，-1 为忽略
        nsp_labels: torch.Tensor,     # [B]     0=IsNext, 1=NotNext
    ) -> BertLossOutput:
        """
        lm_labels:  [B, T]，-100 或 -1 位置不参与 loss
        nsp_labels: [B]，二值标签
        """
        _dbg("LOSS_LM_LABELS", lm_labels)
        _dbg("LOSS_NSP_LABELS", nsp_labels)

        # MLM loss：reshape 到 [B*T, V] 后计算
        B, T, V = bert_output.mlm_logits.shape
        mlm_logits_flat = bert_output.mlm_logits.view(B * T, V)
        lm_labels_flat = lm_labels.view(B * T).long()

        # 上游使用 -1 或 -100 作为 ignore_index；Walpurgis 统一到 -100
        # 若标签含 -1，做一次映射（兼容上游旧格式）
        if (lm_labels_flat == -1).any():
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat == -1, -100)
            _dbg("LOSS_LABEL_REMAP", "mapped -1 → -100 (legacy compat)")

        mlm_loss = F.cross_entropy(mlm_logits_flat, lm_labels_flat, ignore_index=-100)
        _dbg("LOSS_MLM", mlm_loss)

        # NSP loss
        nsp_loss = F.cross_entropy(bert_output.nsp_logits, nsp_labels.long())
        _dbg("LOSS_NSP", nsp_loss)

        total = self.lm_loss_weight * mlm_loss + self.nsp_loss_weight * nsp_loss
        _dbg("LOSS_TOTAL", total)

        # 统计参与 MLM 的 token 数（非 ignore 位置）
        num_tokens = int((lm_labels_flat != -100).sum().item())

        return BertLossOutput(
            total_loss=total,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
            mlm_num_tokens=num_tokens,
        )
