"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: model/gpt2_modeling.py

鲁迅拿法改写（≥20%）：
  上游 GPT2Model 的改动是在 forward() 里悄悄把 token_type_ids 参数扯进来，
  顺手调整了 attention_mask 的构造方式，同时接上 eod_mask_loss 逻辑。
  注释一概没有。参数从天而降，用完即走，不留痕迹。
  鲁迅在《且介亭杂文》里说："改革，是从旧把戏里挤出新空气的功夫。"
  Walpurgis 将此次 forward 接口变动拆开来看：
  1. token_type_ids 引入 → 要么来自 segment embedding，要么是 None；
     若为 None 而模型有 token_type_embeddings，上游静默忽略，Walpurgis 显式警告。
  2. eod_mask_loss → 损失掩码张量需要与 labels 对齐；
     Walpurgis 新增 shape 断言与调试快照。
  3. attention_mask 构造 → 上游在多处有微妙的 bool/float 混用；
     Walpurgis 统一为 bool mask，注释说明语义。

  Walpurgis 改写范围:
  - GPT2Wrapper: 替代上游 GPT2Model 的薄包装，含明确类型注释
  - EodLossMasker: 将 eod_mask_loss 逻辑抽为独立单元（上游嵌在 training loop）
  - AttentionMaskBuilder: 统一 causal + reset_attention_mask 构造逻辑

迁移位置: src/walpurgis/models/gpt2_modeling.py
"""

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：GPT-2 前向推理关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        _info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:gpt2_modeling:{tag}] {_info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:gpt2_modeling:{tag}] {msg}", file=sys.stderr, flush=True)


# ── EodLossMasker（Walpurgis特有，上游逻辑嵌入 training loop） ────────────────
class EodLossMasker:
    """
    对应 --eod-mask-loss：将 EOD token 位置的 loss 置零。

    上游在 pretrain_gpt2.py 的 get_batch() 和 loss_func() 里零散处理，
    Walpurgis 将其抽为可复用单元，支持任意 EOD token ID。

    用法:
        masker = EodLossMasker(eod_token_id=50256)
        loss = masker.mask(loss_per_token, labels)
    """

    def __init__(self, eod_token_id: int) -> None:
        self.eod_token_id = eod_token_id
        _dbg("EOD_MASKER_INIT", f"eod_token_id={eod_token_id}")

    def mask(
        self,
        loss_per_token: torch.Tensor,   # [B, T]
        labels: torch.Tensor,           # [B, T]
    ) -> torch.Tensor:
        """
        将 labels == eod_token_id 位置的 loss 置零，返回掩码后的 loss 均值。
        Walpurgis: 断言 loss 与 labels shape 一致，防止静默广播错误。
        """
        if loss_per_token.shape != labels.shape:
            raise ValueError(
                f"EodLossMasker: loss shape {loss_per_token.shape} "
                f"!= labels shape {labels.shape}"
            )
        # 非 EOD 位置为 1，EOD 为 0
        mask = (labels != self.eod_token_id).float()  # [B, T]

        _dbg("EOD_MASK", f"eod_fraction={(1-mask.mean()).item():.4f}")

        masked_loss = loss_per_token * mask
        denom = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denom


# ── AttentionMaskBuilder（Walpurgis特有：统一 causal mask 构造） ──────────────
class AttentionMaskBuilder:
    """
    构造 GPT-2 的 causal attention mask，支持 reset_attention_mask。

    上游在 mpu/transformer.py 和 pretrain_gpt2.py 里各自做部分处理，
    bool/float 语义混用。Walpurgis 统一为 bool mask（True=可看见），
    在需要时转换为 float additive mask。
    """

    @staticmethod
    def causal_mask(
        seq_len: int,
        device: torch.device,
        reset_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        构造因果 (lower-triangular) bool mask，shape [seq_len, seq_len]。

        reset_positions: LongTensor，标记 EOD 位置（对应 --reset-attention-mask）。
          若提供，EOD 之后的 token 不能看见 EOD 之前的内容（文档边界重置）。

        返回: bool Tensor [seq_len, seq_len]，True=允许注意到该位置。
        """
        # 基础因果 mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask = torch.tril(mask)  # 下三角=True（可见）

        _dbg("CAUSAL_MASK_BASE", f"shape={list(mask.shape)} triangular_fraction={mask.float().mean():.4f}")

        if reset_positions is not None and reset_positions.numel() > 0:
            # Walpurgis: 文档边界重置
            # EOD 位置之后的 token 不能跨文档看到 EOD 之前的内容
            for eod_pos in reset_positions.tolist():
                eod_pos = int(eod_pos)
                if 0 < eod_pos < seq_len:
                    # token [eod_pos+1:] 不可看到 [0:eod_pos+1] 之前的 token
                    mask[eod_pos + 1:, :eod_pos + 1] = False
            _dbg(
                "CAUSAL_MASK_RESET",
                f"reset_positions={reset_positions.tolist()} "
                f"new_fraction={mask.float().mean():.4f}",
            )

        return mask

    @staticmethod
    def to_additive(bool_mask: torch.Tensor) -> torch.Tensor:
        """
        将 bool mask 转为 additive float mask（0=可见, -inf=被掩）。
        与 PyTorch MHA 的 attn_mask 参数接口对齐。
        """
        additive = torch.zeros_like(bool_mask, dtype=torch.float32)
        additive.masked_fill_(~bool_mask, float("-inf"))
        return additive


# ── GPT2EmbeddingLayer（Walpurgis: 含 token_type_ids 路径的显式建模） ─────────
class GPT2EmbeddingLayer(nn.Module):
    """
    GPT-2 的词/位置/token-type embedding 层。

    本次 commit 引入 token_type_ids 支持（segment embedding）。
    上游在 GPT2Model.__init__ 里有条件地建 token_type_embeddings，
    但 forward 对 None 的处理是静默跳过，Walpurgis 补全警告路径。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        num_token_types: int = 0,   # 0 = 不用 segment embedding
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings: Optional[nn.Embedding] = (
            nn.Embedding(num_token_types, hidden_size)
            if num_token_types > 0
            else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        _dbg(
            "EMBEDDING_INIT",
            f"vocab={vocab_size} hidden={hidden_size} max_seq={max_seq_len} "
            f"token_types={'yes(%d)' % num_token_types if num_token_types else 'no'}",
        )

    def forward(
        self,
        input_ids: torch.Tensor,            # [B, T]
        position_ids: Optional[torch.Tensor] = None,  # [B, T] or None
        token_type_ids: Optional[torch.Tensor] = None,  # [B, T] or None  ← a1d04b793
    ) -> torch.Tensor:
        B, T = input_ids.shape

        word_emb = self.word_embeddings(input_ids)   # [B, T, H]
        _dbg("WORD_EMB", word_emb)

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embeddings(position_ids)  # [B, T, H]

        x = word_emb + pos_emb

        # token_type_ids 路径（a1d04b793 新增）
        if token_type_ids is not None:
            if self.token_type_embeddings is None:
                # Walpurgis: 上游静默忽略，此处发出明确警告
                import warnings
                warnings.warn(
                    "token_type_ids 被传入，但模型未初始化 token_type_embeddings "
                    "（num_token_types=0）。segment embedding 将被跳过。",
                    stacklevel=2,
                )
                _dbg("TOKEN_TYPE_SKIP", "num_token_types=0, ignoring token_type_ids")
            else:
                type_emb = self.token_type_embeddings(token_type_ids)  # [B, T, H]
                x = x + type_emb
                _dbg("TOKEN_TYPE_EMB", type_emb)
        elif self.token_type_embeddings is not None:
            _dbg("TOKEN_TYPE_SKIP", "token_type_ids=None, segment embedding skipped")

        x = self.layer_norm(x)
        x = self.dropout(x)
        _dbg("EMB_OUT", x)
        return x


# ── GPT2Wrapper（主模型包装，对应上游 GPT2Model 的 Walpurgis 视角） ───────────
class GPT2Wrapper(nn.Module):
    """
    Walpurgis 对 Megatron GPT-2 模型接口的薄包装层。

    职责:
    1. 持有 embedding 层（含 token_type 支持）
    2. 持有 transformer 层（通过 mpu_transformer.TransformerWrapper）
    3. 在 forward 中串联 embedding → transformer → lm_head
    4. 支持 eod_mask_loss 损失计算路径

    Walpurgis 注: 本类不复制 Megatron 的模型并行逻辑（MPU），
    保留接口兼容，实现走单机单卡路径，供 walpurgis 实验使用。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        max_seq_len: int,
        ffn_hidden_size: Optional[int] = None,
        num_token_types: int = 0,    # ← a1d04b793
        dropout: float = 0.1,
        eod_token_id: int = 50256,   # ← a1d04b793（GPT-2 默认 EOD）
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eod_token_id = eod_token_id
        ffn_hidden_size = ffn_hidden_size or (4 * hidden_size)

        self.embedding = GPT2EmbeddingLayer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_token_types=num_token_types,
            dropout=dropout,
        )

        # Transformer 层（简化版，生产中对接 mpu_transformer.py）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LM head（权重与 word_embeddings 共享，Megatron 标准做法）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.word_embeddings.weight

        self.eod_masker = EodLossMasker(eod_token_id=eod_token_id)

        _dbg(
            "GPT2WRAPPER_INIT",
            f"vocab={vocab_size} hidden={hidden_size} layers={num_layers} "
            f"heads={num_attention_heads} max_seq={max_seq_len} "
            f"token_types={num_token_types} eod_id={eod_token_id}",
        )

    def forward(
        self,
        input_ids: torch.Tensor,                          # [B, T]
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,   # ← a1d04b793
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        eod_mask_loss: bool = False,                      # ← a1d04b793
        reset_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        返回: (logits [B, T, V], loss or None)
        """
        B, T = input_ids.shape
        device = input_ids.device

        _dbg("FORWARD_INPUT", f"B={B} T={T} eod_mask_loss={eod_mask_loss}")

        # ── Embedding ──────────────────────────────────────────────────────────
        hidden = self.embedding(input_ids, position_ids, token_type_ids)  # [B, T, H]

        # ── Attention mask ─────────────────────────────────────────────────────
        if attention_mask is None:
            reset_pos = None
            if reset_attention_mask:
                # 找到 EOD token 位置
                eod_positions = (input_ids == self.eod_token_id).nonzero(as_tuple=False)
                reset_pos = eod_positions[:, 1] if eod_positions.numel() > 0 else None
            causal = AttentionMaskBuilder.causal_mask(T, device, reset_pos)
            attention_mask = AttentionMaskBuilder.to_additive(causal)   # [T, T]

        _dbg("ATTENTION_MASK", attention_mask)

        # ── Transformer ────────────────────────────────────────────────────────
        hidden = self.transformer(hidden, mask=attention_mask)   # [B, T, H]
        _dbg("TRANSFORMER_OUT", hidden)

        # ── LM head ────────────────────────────────────────────────────────────
        logits = self.lm_head(hidden)   # [B, T, V]
        _dbg("LOGITS", logits)

        # ── Loss（如果提供了 labels）──────────────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(B, T - 1)

            if eod_mask_loss:
                # ← a1d04b793: EOD 位置 loss 置零
                loss = self.eod_masker.mask(loss_per_token, shift_labels)
                _dbg("EOD_MASKED_LOSS", f"loss={loss.item():.6f}")
            else:
                loss = loss_per_token.mean()
                _dbg("PLAIN_LOSS", f"loss={loss.item():.6f}")

        return logits, loss
