"""Eclipse inherent: GRU+LayerNorm, Transformer+residual."""
import torch, torch.nn as nn
from torch.nn import MultiheadAttention
import sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim; self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout); self.ln = nn.LayerNorm(hidden_dim)
    def forward(self, X):
        B, S, N, H = X.shape
        X = X.transpose(1, 2).reshape(B * N, S, H)
        hx = torch.zeros_like(X[:, 0, :]); output = []
        for t in range(X.shape[1]):
            hx = self.gru_cell(X[:, t, :], hx)
            hx = self.ln(hx)  # LayerNorm after GRU
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        if _ECL_DBG: print(f"[ECL:rnn] hidden_mean={hx.mean().item():.4f} std={hx.std().item():.4f}", file=sys.stderr)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mhsa = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X, K, V):
        attn_out = self.mhsa(X, K, V)[0]
        attn_out = self.dropout(attn_out)
        out = X + attn_out  # Residual connection (vs upstream no residual)
        if _ECL_DBG: print(f"[ECL:transformer] attn_mean={attn_out.mean().item():.4f}", file=sys.stderr)
        return out
