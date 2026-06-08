"""Vortex inherent model: MinGRU + Transformer with relative positional bias.
MinGRU: simplified GRU with only a single gate (forget gate), reducing parameters.
Relative positional bias: learnable bias table indexed by relative position,
avoiding the need for absolute positional encoding in attention computation.
Unlike upstream (standard GRU + vanilla MHA) and eclipse (GRU+LayerNorm + residual MHA)."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class MinGRUCell(nn.Module):
    """Minimal GRU: single gate design for efficiency.
    h_t = (1-z_t) * h_{t-1} + z_t * h_tilde
    where z_t = sigmoid(W_z x_t + U_z h_{t-1})
    h_tilde = tanh(W_h x_t)  -- no reset gate needed
    This reduces parameters by ~33% compared to standard GRU."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gate
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Candidate
        self.W_h = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        h_tilde = torch.tanh(self.W_h(x))
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next

class RNNLayer(nn.Module):
    """Vortex RNN: MinGRU + GroupNorm for stable recurrence."""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mingru_cell = MinGRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        n_groups = min(4, hidden_dim)
        self.norm = nn.GroupNorm(n_groups, hidden_dim)

    def forward(self, X):
        B, S, N, H = X.shape
        X = X.transpose(1, 2).reshape(B * N, S, H)
        hx = torch.zeros_like(X[:, 0, :]); output = []
        for t in range(X.shape[1]):
            hx = self.mingru_cell(X[:, t, :], hx)
            # GroupNorm after MinGRU (need [B*N, H] -> [B*N, H, 1] for GN)
            hx_normed = self.norm(hx.unsqueeze(-1)).squeeze(-1)
            output.append(hx_normed)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        if _VX_DBG:
            print(f"[VX:mingru@inh_model] hidden_mean={hx.mean().item():.4f} std={hx.std().item():.4f}", file=sys.stderr)
        return output

class TransformerLayer(nn.Module):
    """Vortex Transformer with learnable relative positional bias.
    Instead of absolute PE added to tokens, we add a learnable bias to attention
    scores based on relative positions. This is translation-invariant and
    generalizes better to different sequence lengths."""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Learnable relative position bias table: max 64 relative positions
        self.max_rel_pos = 64
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * self.max_rel_pos + 1))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def _get_rel_bias(self, seq_len_q, seq_len_k):
        """Compute relative position bias matrix."""
        q_pos = torch.arange(seq_len_q, device=self.rel_pos_bias.device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=self.rel_pos_bias.device).unsqueeze(0)
        rel_pos = (q_pos - k_pos).clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        bias = self.rel_pos_bias[:, rel_pos.long()]  # [num_heads, seq_q, seq_k]
        return bias

    def forward(self, X, K, V):
        """X: [S, B*N, D], K: [S_k, B*N, D], V: [S_v, B*N, D]"""
        S_q, BN, D = X.shape
        S_k = K.shape[0]
        Q = self.wq(X).view(S_q, BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K_proj = self.wk(K).view(S_k, BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V_proj = self.wv(V).view(S_k, BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # Scaled dot-product attention + relative position bias
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale
        # Add relative position bias
        rel_bias = self._get_rel_bias(S_q, S_k)  # [num_heads, S_q, S_k]
        attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_proj)
        out = out.permute(2, 0, 1, 3).reshape(S_q, BN, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        # Residual connection
        out = X + out
        if _VX_DBG:
            print(f"[VX:transformer@inh_model] attn_mean={attn.mean().item():.4f} rel_bias_range=[{self.rel_pos_bias.min().item():.4f},{self.rel_pos_bias.max().item():.4f}]", file=sys.stderr)
        return out
