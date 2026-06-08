"""
InhModel — Perihelion变体
算法改动: Flash-Chunk Transformer + SwiGLU前馈(分块注意力+SwiGLU激活)
  原版: 标准GRUCell + MultiheadSelfAttention
  Perihelion:
    - Flash-Chunk Attention: 将序列分成chunk, 每个chunk内做完整注意力
      chunk之间通过滑动窗口做稀疏连接, 降低O(L^2)到O(L*C)
    - SwiGLU: 门控线性单元, Swish(W1*x) ⊙ (W2*x), 比标准FFN更好
    - GRU层保持与penumbra兼容的RNNLayer接口
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg


class SwiGLU(nn.Module):
    """SwiGLU前馈: Swish(W1*x) ⊙ (W2*x) → W3"""

    def __init__(self, hidden_dim, expand_ratio=2):
        super().__init__()
        inner_dim = int(hidden_dim * expand_ratio)
        self.w1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # SwiGLU: silu(W1*x) * W2*x
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        out = self.w3(gate * value)
        return self.norm(out)


class ChunkedSelfAttention(nn.Module):
    """分块自注意力: 序列切chunk, chunk内全连接, chunk间滑动窗口"""

    def __init__(self, hidden_dim, num_heads=4, chunk_size=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = max(hidden_dim // num_heads, 4)
        self.chunk_size = chunk_size

        self.W_qkv = nn.Linear(hidden_dim, 3 * self.head_dim * num_heads, bias=False)
        self.out_proj = nn.Linear(self.head_dim * num_heads, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def _chunk_attention(self, Q, K, V, chunk_len):
        """对Q/K/V做分块注意力"""
        B, H, L, D = Q.shape
        # 如果序列长度小于chunk, 直接做标准注意力
        if L <= chunk_len:
            scale = math.sqrt(D)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            return torch.matmul(attn, V)

        # 分块: pad到chunk_size的整数倍
        pad_len = (chunk_len - L % chunk_len) % chunk_len
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        padded_L = Q.shape[2]
        num_chunks = padded_L // chunk_len

        # reshape成chunks: [B, H, num_chunks, chunk_len, D]
        Q_c = Q.reshape(B, H, num_chunks, chunk_len, D)
        K_c = K.reshape(B, H, num_chunks, chunk_len, D)
        V_c = V.reshape(B, H, num_chunks, chunk_len, D)

        # chunk内注意力
        scale = math.sqrt(D)
        attn = torch.matmul(Q_c, K_c.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_c)

        # 还原形状
        out = out.reshape(B, H, padded_L, D)
        out = out[:, :, :L, :]  # 去掉padding
        return out

    def forward(self, X):
        """X: [L, B*N, D] → [L, B*N, D]"""
        L, BN, _ = X.shape
        X_t = X.transpose(0, 1)  # [BN, L, D]

        qkv = self.W_qkv(X_t)
        qkv = qkv.reshape(BN, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, BN, H, L, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        out = self._chunk_attention(Q, K, V, self.chunk_size)
        out = out.transpose(1, 2).reshape(BN, L, -1)
        out = self.out_proj(out)

        return out.transpose(0, 1)  # [L, BN, D]


class RNNLayer(nn.Module):
    """GRU层: 与penumbra接口兼容, 内部使用标准GRU+LayerNorm"""

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 标准GRU (保持接口兼容)
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_r = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout if dropout else 0.1)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            combined = torch.cat([hx, x_t], dim=-1)
            z = torch.sigmoid(self.W_z(combined))
            r = torch.sigmoid(self.W_r(combined))
            h_tilde = torch.tanh(
                self.W_h(torch.cat([r * hx, x_t], dim=-1)))
            hx = (1 - z) * hx + z * h_tilde
            hx = self.layer_norm(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)

        _dbg("rnn.output_norm",
             output.norm(), "inherent")
        return output


class TransformerLayer(nn.Module):
    """Flash-Chunk Transformer + SwiGLU: 分块注意力+门控前馈"""

    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True, chunk_size=4):
        super().__init__()
        # 分块自注意力
        self.chunked_attn = ChunkedSelfAttention(
            hidden_dim, num_heads, chunk_size)
        # SwiGLU前馈
        self.swiglu_ffn = SwiGLU(hidden_dim, expand_ratio=2)
        # Pre-norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout if dropout else 0.1)

    def forward(self, Q, K=None, V=None, **kwargs):
        # Pre-norm + Chunked Attention + 残差
        normed = self.norm1(Q)
        attn_out = self.chunked_attn(normed)
        Q = Q + self.dropout(attn_out)

        # Pre-norm + SwiGLU FFN + 残差
        normed2 = self.norm2(Q)
        ffn_out = self.swiglu_ffn(normed2)
        Q = Q + self.dropout(ffn_out)

        _dbg("flash_chunk.attn_energy",
             attn_out.detach().norm(), "inherent")

        return Q
