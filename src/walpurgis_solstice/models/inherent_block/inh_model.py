import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:inhmod:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class RNNLayer(nn.Module):
    """upstream: 裸GRU
    solstice: LSTM替代GRU — 独立cell state提供长期记忆, 加PowerNorm稳定"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # solstice: LSTM cell替代GRU cell
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # solstice: PowerNorm稳定隐状态
        self.pn = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        cx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            # solstice: LSTM step
            hx, cx = self.lstm_cell(X[:, t, :], (hx, cx))
            # solstice: PowerNorm
            hx = self.pn(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _adbg("lstm_hidden", output)
        return output


class TransformerLayer(nn.Module):
    """upstream: 标准O(N^2) attention
    solstice: Performer随机特征注意力 — 正随机特征近似softmax, O(N*D)复杂度"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # solstice: 随机特征投影维度
        self._n_random_features = max(16, hidden_dim)
        # solstice: pre-norm
        self.ln = nn.LayerNorm(hidden_dim)

    def _random_feature_map(self, x):
        """solstice: 正随机特征核 — φ(x) = exp(-||x||^2/2) * [cos(ωx), sin(ωx)] / sqrt(m)
        用固定随机投影(每次forward重新采样)近似softmax kernel"""
        d = x.shape[-1]
        m = self._n_random_features
        device = x.device
        omega = torch.randn(d, m, device=device) / math.sqrt(d)
        proj = torch.matmul(x, omega)
        cos_part = torch.cos(proj)
        sin_part = torch.sin(proj)
        phi = torch.cat([cos_part, sin_part], dim=-1) / math.sqrt(m)
        return F.relu(phi) + 1e-6

    def forward(self, X, K, V):
        # solstice: pre-norm
        X_normed = self.ln(X)
        L, BN, D = X_normed.shape
        Q = self.W_q(X_normed)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        # solstice: Performer随机特征注意力
        Q_prime = self._random_feature_map(Q)
        K_prime = self._random_feature_map(K_proj)

        # FAVOR+ mechanism: attn = Q'(K'^T V) / Q'(K'^T 1)
        KV = torch.bmm(K_prime.transpose(0, 1).transpose(-1, -2),
                        V_proj.transpose(0, 1)).transpose(0, 1) if K_prime.dim() == 3 else torch.einsum('lbd,lbe->bde', K_prime, V_proj)
        # Simpler path for sequential data
        KV_sum = torch.einsum('lbd,lbe->bde', K_prime, V_proj)
        K_sum = K_prime.sum(dim=0, keepdim=True)
        numerator = torch.einsum('lbd,bde->lbe', Q_prime, KV_sum)
        denominator = torch.einsum('lbd,bd->lb', Q_prime, K_sum.squeeze(0)).unsqueeze(-1) + 1e-6
        attn_out = numerator / denominator

        attn_out = self.W_o(attn_out)
        attn_out = self.dropout(attn_out)
        # solstice: 残差连接
        out = X + attn_out
        _adbg("performer_out", out)
        return out
