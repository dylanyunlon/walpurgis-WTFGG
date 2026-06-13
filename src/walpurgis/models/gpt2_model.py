"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/model/gpt2_model.py（新增文件，119行）

鲁迅拿法改写（≥20%）：
  上游 gpt2_model.py 是一个薄包装（thin wrapper）：
  GPT2Model.__init__ 里调用 get_language_model()，
  forward() 里对 LanguageModelCore 的输出加一个 word embedding weight 的反投影，
  得到 logits。119行里至少有 30 行是导入和注释空行。
  这种「封装」，鲁迅说过：「挂羊头，卖狗肉——GPT2Model 其实就是 TransformerLM 加一个 Linear。」

  Walpurgis 改写要点：
  1. GPT2OutputHead: 将上游 gpt2_model.py 里的 lm_head（weight-tied 投影）
     单独封装，与 MaskedLMHead 对称命名。
  2. GPT2ModelWrapper: 持有 LanguageModelCore + GPT2OutputHead，
     forward() 返回 GPT2Output dataclass。
  3. GPT2LossComputer: 将 pretrain_gpt2.py 里的 CrossEntropy loss 抽出，
     与 BertLossComputer 对称（上游两套 pretrain 脚本各自算 loss，Walpurgis 统一接口）。
  4. tokentype_ids 透传：上游 GPT2Model.forward 新增了 tokentype_ids 参数，
     Walpurgis 显式保留，并在为 None 时加 _dbg 说明。

迁移位置: src/walpurgis/models/gpt2_model.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：gpt2_model 前向关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:gpt2_model:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:gpt2_model:{tag}] {msg}", file=sys.stderr, flush=True)


# ── GPT2OutputHead ────────────────────────────────────────────────────────────
# 上游：lm_logits = F.linear(hidden_states, word_embeddings_weight)
# 即 weight-tied 的 un-embedding 投影（无 bias）。
# Walpurgis：封装为 GPT2OutputHead，weight tying 显式调用。

class GPT2OutputHead(nn.Module):
    """
    GPT-2 语言模型输出头（weight-tied un-embedding）。

    上游实现：
        output = torch.matmul(hidden_states, self.word_embeddings_weight.t())
    Walpurgis：用 nn.Linear(bias=False) 持有权重引用，tie_weights() 完成绑定。
    等价语义，但模块化更清晰。
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        # 只定义 Linear，权重将由 tie_weights 绑定到 embedding
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        _dbg("GPT2_HEAD_INIT", f"hidden={hidden_size}, vocab={vocab_size}")

    def tie_weights(self, word_embedding_weight: torch.Tensor) -> None:
        """
        将 decoder.weight 与 word_embeddings.weight 绑定。
        上游：直接 self.word_embeddings_weight = model.embedding.word_embeddings.weight
        Walpurgis：显式调用，记录 shape 断言。
        """
        assert self.decoder.weight.shape == word_embedding_weight.shape, (
            f"GPT2 weight tying shape mismatch: "
            f"decoder={self.decoder.weight.shape}, "
            f"word_emb={word_embedding_weight.shape}"
        )
        self.decoder.weight = word_embedding_weight
        _dbg("GPT2_WEIGHT_TIED", f"shape={list(word_embedding_weight.shape)}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        returns:       [B, T, V]  (语言模型 logits)
        """
        _dbg("GPT2_HEAD_INPUT", hidden_states)
        logits = self.decoder(hidden_states)
        _dbg("GPT2_HEAD_LOGITS", logits)
        return logits


# ── GPT2Output dataclass ──────────────────────────────────────────────────────
@dataclass
class GPT2Output:
    """GPT2ModelWrapper.forward 的结构化返回值"""
    logits: torch.Tensor            # [B, T, V]
    hidden_states: torch.Tensor     # [B, T, H]  最终 transformer 输出


# ── GPT2ModelWrapper ──────────────────────────────────────────────────────────
# 替代上游 GPT2Model，持有 LanguageModelCore + GPT2OutputHead。

class GPT2ModelWrapper(nn.Module):
    """
    GPT-2 全模型（Walpurgis 版）。

    对应上游 GPT2Model（megatron/model/gpt2_model.py）。
    上游 GPT2Model.forward() 接收 tokentype_ids（本次 commit 新增），
    Walpurgis 透传并在 _dbg 里标注其来源语义。

    关键变化（vs 上游 gpt2_modeling.py 旧版）：
      - 不再继承 MegatronModule，改为普通 nn.Module（单机版）
      - tokentype_ids 参数显式支持（旧版 gpt2_modeling.py 无此参数）
      - weight tying 通过 GPT2OutputHead.tie_weights() 显式完成
    """

    def __init__(
        self,
        language_model,     # LanguageModelCore(add_pooler=False)
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model

        self.output_head = GPT2OutputHead(hidden_size, vocab_size)
        # weight tying：上游在 GPT2Model.__init__ 末尾执行
        word_emb_weight = language_model.embedding.word_embeddings.weight
        self.output_head.tie_weights(word_emb_weight)

        _dbg("GPT2_WRAPPER_INIT",
             f"hidden={hidden_size}, vocab={vocab_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> GPT2Output:
        """
        input_ids:      [B, T]
        position_ids:   [B, T]
        attention_mask: [B, 1, T, T]  (causal mask，由调用者构造)
        tokentype_ids:  [B, T] or None  （本次 commit 73af12903 新增支持）
        labels:         忽略（loss 由 GPT2LossComputer 计算）

        返回：GPT2Output(logits, hidden_states)
        """
        if tokentype_ids is not None:
            _dbg("FWD_TOKENTYPE", f"tokentype_ids provided: shape={list(tokentype_ids.shape)}")
        else:
            _dbg("FWD_TOKENTYPE", "tokentype_ids=None (standard GPT-2 mode)")

        _dbg("FWD_INPUT_IDS", input_ids)
        _dbg("FWD_POSITION_IDS", position_ids)
        _dbg("FWD_ATTN_MASK", attention_mask)

        # LanguageModelCore with add_pooler=False → hidden_states only
        hidden_states = self.language_model(
            input_ids, position_ids, attention_mask, tokentype_ids
        )
        _dbg("FWD_HIDDEN", hidden_states)

        logits = self.output_head(hidden_states)
        _dbg("FWD_LOGITS", logits)

        return GPT2Output(logits=logits, hidden_states=hidden_states)


# ── GPT2LossComputer ──────────────────────────────────────────────────────────
# 上游 pretrain_gpt2.py 的 loss_func() 里：
#   losses = mpu.vocab_parallel_cross_entropy(output_tensor.contiguous().float(), labels)
#   loss_mask = loss_mask.view(-1).float()
#   loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
# Walpurgis：去掉 mpu 依赖，等价单机实现。

@dataclass
class GPT2LossOutput:
    """GPT2LossComputer 的结构化返回"""
    loss: torch.Tensor          # 加权平均 loss（标量）
    num_tokens: int             # 参与 loss 计算的 token 数（loss_mask 非零位置）


class GPT2LossComputer:
    """
    GPT-2 语言模型损失计算（对应上游 pretrain_gpt2.py loss_func）。

    loss_mask 语义：
        1.0 → 参与 loss 计算（正常 token）
        0.0 → 忽略（EOD token 后的位置，当 --eod-mask-loss 启用时）

    上游用 mpu.vocab_parallel_cross_entropy，Walpurgis 用标准 F.cross_entropy，
    语义等价（单机场景无需并行 all-reduce）。
    """

    def __call__(
        self,
        gpt2_output: GPT2Output,
        labels: torch.Tensor,       # [B, T]  下一个 token 的 id
        loss_mask: torch.Tensor,    # [B, T]  1.0 参与，0.0 忽略
    ) -> GPT2LossOutput:
        """
        labels:    [B, T]  通常是 input_ids shifted by 1
        loss_mask: [B, T]  float tensor，0/1
        """
        _dbg("GPT2_LOSS_LABELS", labels)
        _dbg("GPT2_LOSS_MASK", loss_mask)

        B, T, V = gpt2_output.logits.shape
        logits_flat = gpt2_output.logits.view(B * T, V).float()
        labels_flat = labels.view(B * T).long()

        # per-token loss，不做 reduction
        per_token_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")
        _dbg("GPT2_PER_TOKEN_LOSS", per_token_loss)

        # 加权平均：loss_mask 为 0 的位置不参与
        mask_flat = loss_mask.view(B * T).float()
        masked_loss = per_token_loss * mask_flat
        num_tokens = int(mask_flat.sum().item())

        if num_tokens == 0:
            _dbg("GPT2_LOSS_WARN", "loss_mask all zeros—loss will be 0")
            loss = masked_loss.sum() * 0.0
        else:
            loss = masked_loss.sum() / mask_flat.sum()

        _dbg("GPT2_LOSS_TOTAL", loss)
        return GPT2LossOutput(loss=loss, num_tokens=num_tokens)
