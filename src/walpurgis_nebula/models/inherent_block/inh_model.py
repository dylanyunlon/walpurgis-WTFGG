"""Nebula inherent: IndRNN (independently recurrent) + flash attention pattern."""
import torch, torch.nn as nn, math
from torch.nn import MultiheadAttention
import sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class IndRNNCell(nn.Module):
    """Independently Recurrent Neural Network cell.
    h_t = act(W_x * x_t + w_h ⊙ h_{t-1} + b)
    where w_h is a vector (element-wise recurrence), not a matrix.
    This prevents vanishing/exploding gradients by bounding recurrent weights."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_x = nn.Linear(input_dim, hidden_dim)
        # Element-wise recurrent weight (key IndRNN innovation)
        self.w_h = nn.Parameter(torch.Tensor(hidden_dim))
        self.activation = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # Initialize recurrent weight to allow long-range dependencies
        nn.init.uniform_(self.w_h, -1.0, 1.0)

    def forward(self, x, hx=None):
        """x: [batch, input_dim], hx: [batch, hidden_dim]"""
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        # IndRNN: element-wise multiply instead of matrix multiply
        return self.activation(self.W_x(x) + self.w_h * hx)


class RNNLayer(nn.Module):
    """IndRNN layer replacing upstream GRU."""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ind_rnn_cell = IndRNNCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            hx = self.ind_rnn_cell(X[:, t, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        if _NEB_DBG:
            print(f"[NEB:indrnn@inh_model] seq_len={seq_len} out_norm={output.norm().item():.4f}", file=sys.stderr)
        return output


class FlashAttentionPattern(nn.Module):
    """Flash-attention-style chunked attention pattern.
    Splits Q,K,V into chunks and computes attention block-wise for memory efficiency.
    Uses standard PyTorch ops (no custom CUDA kernels) but follows the tiling logic."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0, bias=True, chunk_size=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _chunked_attention(self, Q, K, V):
        """Chunked attention: process in blocks to simulate flash attention tiling."""
        S_q, B, H, D = Q.shape[0], Q.shape[1], self.num_heads, self.head_dim
        S_kv = K.shape[0]
        # Reshape: [S, B, H*D] -> [S, B, H, D]
        Q = Q.view(S_q, B, H, D)
        K = K.view(S_kv, B, H, D)
        V = V.view(S_kv, B, H, D)
        # Transpose for attention: [B, H, S, D]
        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)
        # Chunked computation
        cs = min(self.chunk_size, S_q)
        outputs = []
        for i in range(0, S_q, cs):
            q_chunk = Q[:, :, i:i+cs, :]  # [B, H, cs, D]
            # Attend to all keys (full attention, but query-chunked)
            attn = torch.matmul(q_chunk, K.transpose(-1, -2)) * self.scale  # [B, H, cs, S_kv]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out_chunk = torch.matmul(attn, V)  # [B, H, cs, D]
            outputs.append(out_chunk)
        out = torch.cat(outputs, dim=2)  # [B, H, S_q, D]
        out = out.permute(2, 0, 1, 3).reshape(S_q, B, H * D)  # [S_q, B, H*D]
        return out

    def forward(self, X, K_in, V_in):
        Q = self.W_q(X)
        K = self.W_k(K_in)
        V = self.W_v(V_in)
        out = self._chunked_attention(Q, K, V)
        out = self.out_proj(out)
        out = self.dropout(out)
        if _NEB_DBG:
            print(f"[NEB:flash_attn@inh_model] S={X.shape[0]} out_norm={out.norm().item():.4f}", file=sys.stderr)
        return out


TransformerLayer = FlashAttentionPattern
