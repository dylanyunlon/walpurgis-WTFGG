import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

# Delta vs upstream:
#   1. GRUCell → LSTMCell (stronger gating, forget gate helps long sequences)
#   2. TransformerLayer adds residual + LayerNorm (pre-norm style)

class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # ── delta 1: LSTM instead of GRU ──
        self.rnn_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros(B * N, D, device=X.device, dtype=X.dtype)
        cx = torch.zeros(B * N, D, device=X.device, dtype=X.dtype)
        output = []
        for t in range(L):
            hx, cx = self.rnn_cell(X[:, t, :], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=0)     # [L, B*N, D]
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mhsa = MultiheadAttention(hidden_dim, num_heads,
                                       dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # ── delta 2: pre-norm residual ──
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        normed = self.ln(X)                         # delta 2
        attn_out = self.mhsa(normed, K, V)[0]
        attn_out = self.dropout(attn_out)
        return X + attn_out                         # delta 2: residual
