"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/model/transformer.py（新增文件，490行）

鲁迅拿法改写（≥20%）：
  上游把 mpu/transformer.py（647行）拆解、精简，提取出不依赖 mpu 并行的
  单机单卡 transformer 实现（490行），作为 language_model.py 的骨干。
  原 mpu/transformer.py 保留做分布式版本。
  此次重构的核心矛盾：上游用「把代码挪个地方」充当「重构」，
  内部实现几乎原封不动，连变量名都懒得改，只是 import 路径变了。
  鲁迅说：「换了招牌，货色还是那一套。」

  Walpurgis 改写要点：
  1. SelfAttentionCore: 将 ParallelSelfAttention 的串行路径单独封装，
     QKV 投影、scaled dot-product、输出投影各为独立方法，便于插桩。
  2. FeedForwardBlock: 将 ParallelMLP 的两层线性+激活抽为 FFN 单元，
     支持 gelu/relu 切换（上游默认 gelu，无法覆盖）。
  3. TransformerLayer: 整合 SelfAttentionCore + FeedForwardBlock，
     PreLN（上游默认）路径显式标注，PostLN 路径预留接口。
  4. TransformerStack: 替代上游 ParallelTransformer 的单机版，
     层级 _dbg 断点（每层前后、attn/ffn 各阶段）。

迁移位置: src/walpurgis/models/transformer.py
"""

import os
import sys
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：transformer 前向关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:transformer:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:transformer:{tag}] {msg}", file=sys.stderr, flush=True)


# ── SelfAttentionCore ─────────────────────────────────────────────────────────
# 上游 ParallelSelfAttention 的串行路径（去掉 tensor parallel 分片）。
# Walpurgis：将 QKV 投影、softmax attention、输出投影分为三个命名方法。

class SelfAttentionCore(nn.Module):
    """
    单机版多头自注意力（无 tensor parallel）。

    对应上游 ParallelSelfAttention 的单卡路径。
    上游将 QKV 投影合并为一个 3*hidden Linear，
    Walpurgis 保留此合并，但拆解 forward() 为三个语义阶段：
      _project_qkv → _compute_attention → _project_output
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float,
        init_method,
        output_layer_init_method,
        attention_mask_func: Callable,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.attention_mask_func = attention_mask_func
        self.scale = math.sqrt(self.head_dim)

        # 上游：self.query_key_value = ColumnParallelLinear(...)
        # Walpurgis：单机 Linear，等价路径
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        init_method(self.qkv_proj.weight)

        # 上游：self.dense = RowParallelLinear(...)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        output_layer_init_method(self.out_proj.weight)

        self.attn_dropout = nn.Dropout(p=attention_dropout)
        _dbg("ATTN_INIT", f"hidden={hidden_size}, heads={num_heads}, head_dim={self.head_dim}")

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple_QKV:
        """QKV 投影 + reshape 到 multi-head 格式"""
        B, T, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)  # [B, T, 3*H]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)         # 各 [B, T, heads, head_dim]
        # 转为 [B, heads, T, head_dim]（上游的 b n s np 格式）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        _dbg("QKV_PROJ", f"q.shape={list(q.shape)}")
        return q, k, v

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scaled dot-product attention，含 mask 与 dropout"""
        # [B, heads, T, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        _dbg("ATTN_SCORES_PRE_MASK", attn_scores)

        if attention_mask is not None:
            # 上游 attention_mask_func 约定：mask=True 位置填充 -10000.0
            attn_scores = self.attention_mask_func(attn_scores, attention_mask)
        _dbg("ATTN_SCORES_POST_MASK", attn_scores)

        attn_weights = F.softmax(attn_scores, dim=-1)
        # 上游在 softmax 后 dropout（commit a1d04b793 迁移过来的细节）
        attn_weights = self.attn_dropout(attn_weights)
        _dbg("ATTN_WEIGHTS", attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, heads, T, head_dim]
        return context

    def _project_output(self, context: torch.Tensor) -> torch.Tensor:
        """multi-head concat → output projection"""
        B, H, T, D = context.shape
        context = context.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.out_proj(context)
        _dbg("ATTN_OUT", out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(hidden_states)
        context = self._compute_attention(q, k, v, attention_mask)
        return self._project_output(context)


# 类型别名（Python 3.8 兼容）
from typing import Tuple as Tuple_QKV  # noqa: E402  (after use, before class ends)


# ── FeedForwardBlock ──────────────────────────────────────────────────────────
# 上游 ParallelMLP：dense_h_to_4h → gelu → dense_4h_to_h。
# Walpurgis：命名为 FeedForwardBlock，激活函数可切换。

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward（上游 ParallelMLP 的单机版）。

    上游 ffn_hidden_size 通常为 4 * hidden_size（BERT/GPT-2 标准）。
    激活函数：上游硬编码 gelu，Walpurgis 支持传入任意 act_fn。
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        hidden_dropout: float,
        init_method,
        output_layer_init_method,
        act_fn: Callable = F.gelu,
    ) -> None:
        super().__init__()
        self.act_fn = act_fn

        # 上游：dense_h_to_4h = ColumnParallelLinear
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        init_method(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # 上游：dense_4h_to_h = RowParallelLinear
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=True)
        output_layer_init_method(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.dropout = nn.Dropout(p=hidden_dropout)
        _dbg("FFN_INIT", f"hidden={hidden_size}, ffn_hidden={ffn_hidden_size}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _dbg("FFN_INPUT", hidden_states)
        x = self.act_fn(self.fc1(hidden_states))
        _dbg("FFN_ACTIVATED", x)
        x = self.dropout(self.fc2(x))
        _dbg("FFN_OUTPUT", x)
        return x


# ── TransformerLayer ──────────────────────────────────────────────────────────
# 上游 ParallelTransformerLayer：PreLN → attn → residual → PreLN → FFN → residual。
# Walpurgis：同一结构，但 PreLN / PostLN 路径用 use_pre_ln 参数显式区分。

class TransformerLayer(nn.Module):
    """
    单层 Transformer（Pre-LayerNorm 默认，PostLN 可选）。

    上游注释：无。
    Walpurgis 标注：上游实现是 Pre-LN（先 LayerNorm 再 attention），
    与原始 BERT 的 Post-LN 不同。本层两种路径均实现，use_pre_ln 控制。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        attention_mask_func: Callable,
        attention_dropout: float,
        hidden_dropout: float,
        init_method,
        output_layer_init_method,
        layer_idx: int = 0,
        use_pre_ln: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_pre_ln = use_pre_ln

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        self.attention = SelfAttentionCore(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            attention_mask_func=attention_mask_func,
        )
        self.ffn = FeedForwardBlock(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_dropout=hidden_dropout,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        _dbg("LAYER_INIT", f"layer_idx={layer_idx}, pre_ln={use_pre_ln}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _dbg(f"LAYER{self.layer_idx}_INPUT", hidden_states)

        if self.use_pre_ln:
            # Pre-LN 路径（上游实现）
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
            attn_out = self.attention(normed, attention_mask)
            _dbg(f"LAYER{self.layer_idx}_ATTN_OUT", attn_out)
            hidden_states = residual + self.hidden_dropout(attn_out)

            residual = hidden_states
            normed = self.post_attention_layernorm(hidden_states)
            ffn_out = self.ffn(normed)
            _dbg(f"LAYER{self.layer_idx}_FFN_OUT", ffn_out)
            hidden_states = residual + ffn_out
        else:
            # Post-LN 路径（原始 BERT 风格，Walpurgis 扩展）
            attn_out = self.attention(hidden_states, attention_mask)
            _dbg(f"LAYER{self.layer_idx}_ATTN_OUT", attn_out)
            hidden_states = self.input_layernorm(hidden_states + self.hidden_dropout(attn_out))
            ffn_out = self.ffn(hidden_states)
            _dbg(f"LAYER{self.layer_idx}_FFN_OUT", ffn_out)
            hidden_states = self.post_attention_layernorm(hidden_states + ffn_out)

        _dbg(f"LAYER{self.layer_idx}_OUTPUT", hidden_states)
        return hidden_states


# ── TransformerStack ──────────────────────────────────────────────────────────
# 上游 ParallelTransformer：N 层 TransformerLayer + 最终 LayerNorm。
# Walpurgis：保留结构，加层级 _dbg，暴露 per-layer hidden states 接口。

class TransformerStack(nn.Module):
    """
    N 层 Transformer 堆叠（上游 ParallelTransformer 的单机版）。

    上游：输出只有最终 hidden_states。
    Walpurgis 扩展：output_all_hidden_states=True 时返回所有层的 hidden_states，
    用于 probing、可视化等分析场景。
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        attention_mask_func: Callable,
        attention_dropout: float,
        hidden_dropout: float,
        init_method,
        output_layer_init_method,
        use_pre_ln: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                ffn_hidden_size=ffn_hidden_size,
                attention_mask_func=attention_mask_func,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_idx=i,
                use_pre_ln=use_pre_ln,
            )
            for i in range(num_layers)
        ])

        # 上游在 ParallelTransformer 末尾有一个 final LayerNorm
        self.final_layernorm = nn.LayerNorm(hidden_size)
        _dbg("STACK_INIT", f"num_layers={num_layers}, hidden={hidden_size}, heads={num_attention_heads}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_all_hidden_states: bool = False,
    ):
        """
        返回：
          output_all_hidden_states=False → final hidden_states [B, T, H]
          output_all_hidden_states=True  → list of [B, T, H] per layer + final
        """
        _dbg("STACK_INPUT", hidden_states)
        all_hidden = [] if output_all_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
            if output_all_hidden_states:
                all_hidden.append(hidden_states)
            _dbg(f"STACK_LAYER{i}_DONE", hidden_states)

        hidden_states = self.final_layernorm(hidden_states)
        _dbg("STACK_FINAL_NORM", hidden_states)

        if output_all_hidden_states:
            all_hidden.append(hidden_states)
            return all_hidden
        return hidden_states
