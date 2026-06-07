import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:inhmod:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)


class RNNLayer(nn.Module):
    """upstream: 裸GRU
    solstice: LSTM替代GRU — 增加cell state提供长程记忆能力"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # solstice: LSTMCell替代GRUCell
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        cx = torch.zeros_like(X[:, 0, :])  # LSTM cell state
        output = []
        for t in range(X.shape[1]):
            hx, cx = self.lstm_cell(X[:, t, :], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _sdbg("lstm_hidden", output)
        return output


def _random_feature_map(x, num_features=None, seed=None):
    """solstice: FAVOR+ 随机特征近似 — Performer式正交随机特征
    将softmax attention O(N²)近似为O(N)的线性attention"""
    d = x.shape[-1]
    if num_features is None:
        num_features = max(d, 64)
    # 生成正交随机投影矩阵
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        W = torch.randn(num_features, d, device=x.device, generator=gen)
    else:
        W = torch.randn(num_features, d, device=x.device)
    # QR正交化提高近似精度
    Q, _ = torch.linalg.qr(W.T)
    W_orth = Q.T[:num_features]  # [num_features, d]
    # FAVOR+: exp(xW^T - ||x||^2/2) 作为正特征映射
    xW = torch.matmul(x, W_orth.T)  # [..., num_features]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
    phi = torch.exp(xW - x_norm_sq)
    return phi


class TransformerLayer(nn.Module):
    """upstream: 标准softmax attention O(N²)
    solstice: FAVOR+ Performer式随机特征近似注意力 O(N)"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # solstice: 随机特征维度
        self.num_random_features = max(self.head_dim, 32)

    def forward(self, X, K, V):
        L, BN, D = X.shape

        Q = self.W_q(X)  # [L, BN, D]
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        # Reshape for multi-head
        Q = Q.view(L, BN, self.num_heads, self.head_dim)
        K_proj = K_proj.view(K_proj.shape[0], BN, self.num_heads, self.head_dim)
        V_proj = V_proj.view(V_proj.shape[0], BN, self.num_heads, self.head_dim)

        # solstice: FAVOR+ 随机特征映射
        Q_prime = _random_feature_map(Q, self.num_random_features)  # [L, BN, H, M]
        K_prime = _random_feature_map(K_proj, self.num_random_features)

        # Linear attention: (Q'(K'^T V)) / (Q' sum(K'))
        # KV = K'^T @ V: [BN, H, M, head_dim]
        K_prime_t = K_prime.permute(1, 2, 3, 0)  # [BN, H, M, Lk]
        V_t = V_proj.permute(1, 2, 0, 3)  # [BN, H, Lk, head_dim]
        KtV = torch.matmul(K_prime_t, V_t)  # [BN, H, M, head_dim]

        Q_prime_r = Q_prime.permute(1, 2, 0, 3)  # [BN, H, L, M]
        attn_out = torch.matmul(Q_prime_r, KtV)  # [BN, H, L, head_dim]

        # Normalizer
        K_sum = K_prime.sum(dim=0)  # [BN, H, M]
        normalizer = torch.matmul(Q_prime_r, K_sum.unsqueeze(-1)) + 1e-6  # [BN, H, L, 1]
        attn_out = attn_out / normalizer

        # Reshape back: [L, BN, D]
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(L, BN, D)
        attn_out = self.W_o(attn_out)
        attn_out = self.dropout(attn_out)

        # Residual
        out = X + attn_out
        _sdbg("performer_out", out)
        return out
