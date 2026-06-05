"""
inh_model.py — CardGame Inherent Model (RNN + Transformer)
算法改写 (vs upstream):
  - ReLU隐式激活 → GELU (更平滑, 在Transformer中表现更好)
  - RNNLayer中使用gradient checkpoint减少显存占用
  - TransformerLayer dropout后加residual connection
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.checkpoint import checkpoint

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="InhModel"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class RNNLayer(nn.Module):
    """CardGame RNNLayer: GRU with GELU preactivation + gradient checkpoint"""

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # CardGame: GELU gate before GRU
        self.pre_gate = nn.GELU()
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)

    def _gru_step(self, x_t, hx):
        """单步GRU, 可被gradient checkpoint包裹"""
        return self.gru_cell(x_t, hx)

    def forward(self, X):
        _dbg("rnn.input", X)
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)

        # CardGame: GELU pre-activation
        X = self.pre_gate(self.pre_proj(X))

        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            # gradient checkpoint减少显存
            if self.training and X.requires_grad:
                hx = checkpoint(self._gru_step, X[:, t, :], hx,
                                use_reentrant=False)
            else:
                hx = self._gru_step(X[:, t, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _dbg("rnn.output", output)
        return output


class TransformerLayer(nn.Module):
    """CardGame TransformerLayer: MSA with residual + GELU FFN"""

    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # CardGame: 后接GELU FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        _dbg("transformer.input", X)
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        # CardGame: residual + FFN + LayerNorm
        out = X + hidden_states_MSA
        out = self.ln(out + self.ffn(out))
        _dbg("transformer.output", out)
        return out
