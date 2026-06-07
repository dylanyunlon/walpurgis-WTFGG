import torch
import torch.nn as nn
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:inhmod:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)


class HighwayGRUCell(nn.Module):
    """equinox: GRU + Highway Connection
    标准GRU输出后加Highway gate: out = T*GRU(x,h) + (1-T)*x
    T = sigmoid(W_T*x + b_T), 可学习的跳过门控
    允许梯度绕过GRU直接回传, 缓解长序列梯度消失"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        # Highway transform gate
        self.highway_gate = nn.Linear(input_dim, hidden_dim)
        # 如果input_dim != hidden_dim, 需要投影
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        nn.init.constant_(self.highway_gate.bias, -1.0)  # 初始偏向carry (跳过)

    def forward(self, x, hx):
        gru_out = self.gru_cell(x, hx)
        T = torch.sigmoid(self.highway_gate(x))
        x_proj = self.input_proj(x)
        # Highway: T * GRU输出 + (1-T) * 输入投影
        out = T * gru_out + (1 - T) * x_proj
        return out


class RNNLayer(nn.Module):
    """upstream: 裸GRU
    equinox: Highway GRU + WeightNorm稳定隐状态"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # equinox: HighwayGRU替代裸GRU
        self.gru_cell = HighwayGRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # equinox: WeightNorm稳定 (通过weight_norm包装一个投影层)
        self.wn_proj = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            hx = self.gru_cell(X[:, t, :], hx)
            hx = self.wn_proj(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _edbg("highway_gru_hidden", output)
        return output


class LinformerAttention(nn.Module):
    """equinox: Linformer低秩投影多头注意力
    将K,V序列长度从N投影到k维 (k<<N), 计算复杂度O(Nk)
    保持多头结构, 每个头独立投影"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, proj_dim=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Linformer: 低秩投影矩阵 E, F
        self.proj_dim = proj_dim
        self.E = nn.Parameter(torch.randn(num_heads, proj_dim, 1) * 0.02)
        self.F = nn.Parameter(torch.randn(num_heads, proj_dim, 1) * 0.02)
        self._seq_len_cached = 0

        self.dropout = nn.Dropout(dropout)

    def _get_proj(self, seq_len, device):
        """动态扩展投影矩阵以匹配序列长度"""
        if seq_len != self._seq_len_cached:
            self._E_expanded = self.E.expand(-1, -1, seq_len).to(device)
            self._F_expanded = self.F.expand(-1, -1, seq_len).to(device)
            # 使用可学习参数通过repeat_interleave扩展
            E_base = self.E[:, :, 0].unsqueeze(-1)  # [heads, proj, 1]
            F_base = self.F[:, :, 0].unsqueeze(-1)
            self._E_expanded = E_base.expand(-1, -1, seq_len).to(device)
            self._F_expanded = F_base.expand(-1, -1, seq_len).to(device)
            self._seq_len_cached = seq_len
        return self._E_expanded, self._F_expanded

    def forward(self, Q, K, V):
        L, BN, D = Q.shape
        Q = self.W_q(Q).view(L, BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = self.W_k(K).view(K.shape[0], BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.W_v(V).view(V.shape[0], BN, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # Q: [BN, heads, L, head_dim]
        # K, V: [BN, heads, S, head_dim]

        S = K.shape[2]
        E, F_mat = self._get_proj(S, Q.device)
        # 投影 K, V: [heads, proj, S] @ [BN, heads, S, head_dim] -> [BN, heads, proj, head_dim]
        E_expanded = E.unsqueeze(0).expand(BN, -1, -1, -1)
        F_expanded = F_mat.unsqueeze(0).expand(BN, -1, -1, -1)
        K_proj = torch.matmul(E_expanded, K)  # [BN, heads, proj, head_dim]
        V_proj = torch.matmul(F_expanded, V)

        # 标准attention, 但K,V只有proj_dim长
        attn = torch.matmul(Q, K_proj.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_proj)
        # [BN, heads, L, head_dim] -> [L, BN, D]
        out = out.permute(2, 0, 1, 3).reshape(L, BN, D)
        out = self.W_o(out)
        _edbg("linformer_mha", out)
        return out


class TransformerLayer(nn.Module):
    """upstream: 裸attention
    equinox: Linformer低秩注意力 + pre-norm残差"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        # equinox: Linformer替代标准MultiheadAttention
        self.linformer_attn = LinformerAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout or 0.1, proj_dim=8)
        self.dropout = nn.Dropout(dropout)
        # equinox: WeightNorm pre-norm
        self.wn_proj = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, X, K, V):
        X_normed = self.wn_proj(X)
        attn_out = self.linformer_attn(X_normed, K, V)
        attn_out = self.dropout(attn_out)
        # 残差连接
        out = X + attn_out
        _edbg("transformer_out", out)
        return out
