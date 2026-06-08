"""
inh_model — Aurora变体
算法改写: Multi-Scale Temporal Processing
  - MultiScaleTemporalLayer替代原始单一GRU序列处理
  - 使用不同kernel size (1, 3, 5) 做temporal pooling
  - 各尺度独立通过线性投影后做cross-scale attention融合
  - TransformerLayer保持标准attention但加gated residual
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class MultiScaleTemporalLayer(nn.Module):
    """Multi-Scale Temporal Processing: 替代原始单一GRU

    使用多个不同kernel size的1D卷积捕捉不同时间尺度的模式,
    然后通过cross-scale attention融合各尺度特征。
    这比单一GRU能更好地捕捉短期波动和长期趋势。
    """
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Aurora: 三个不同尺度的temporal卷积 (kernel=1,3,5)
        # kernel=1: 逐点变换, 捕捉瞬时特征
        # kernel=3: 短程模式
        # kernel=5: 中程趋势
        self.conv_scale1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv_scale3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_scale5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

        # 各尺度投影到统一空间
        self.scale_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # cross-scale attention: 让各尺度互相关注
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=2, dropout=dropout or 0.1, batch_first=False)

        self.dropout = nn.Dropout(dropout or 0.1)
        self.norm = nn.LayerNorm(hidden_dim)

        # 保留一个轻量GRU用于序列位置编码(比PE更灵活)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, X):
        """
        X: [B, L, N, D]
        Returns: [L, B*N, D] (与原始RNNLayer输出格式一致)
        """
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        # reshape到 [B*N, D, L] 用于Conv1d
        X_conv = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        X_conv = X_conv.transpose(1, 2)  # [B*N, D, L]

        # 多尺度temporal pooling
        s1 = self.conv_scale1(X_conv)  # [B*N, D, L]
        s3 = self.conv_scale3(X_conv)  # [B*N, D, L]
        s5 = self.conv_scale5(X_conv)  # [B*N, D, L]

        # 拼接各尺度 → [B*N, 3D, L] → [B*N, L, 3D]
        multi_scale = th.cat([s1, s3, s5], dim=1).transpose(1, 2)

        # 投影到hidden_dim → [B*N, L, D]
        fused = self.scale_proj(multi_scale)
        fused = F.gelu(fused)

        # 转为 [L, B*N, D] 用于cross-scale attention
        fused = fused.transpose(0, 1)

        # cross-scale self-attention: 各时步互相关注
        attn_out, _ = self.cross_attn(fused, fused, fused)
        fused = self.norm(fused + self.dropout(attn_out))

        # 轻量GRU提供序列顺序感知
        X_seq = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X_seq[:, 0, :])
        gru_out = []
        for step in range(seq_len):
            hx = self.gru_cell(X_seq[:, step, :], hx)
            gru_out.append(hx)
        gru_out = th.stack(gru_out, dim=0)  # [L, B*N, D]

        # 融合: multi-scale attention + GRU位置编码
        # 加权组合而非简单相加
        output = 0.7 * fused + 0.3 * gru_out
        output = self.dropout(output)

        return output


# 保留RNNLayer用于forecast分支的兼容性
class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = th.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Aurora: gated residual — 学习残差连接的强度
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(
            X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        # Aurora: gated residual connection
        # gate决定attention输出和原始输入的混合比例
        gate = th.sigmoid(self.gate_linear(X))
        hidden_states_MSA = gate * hidden_states_MSA + (1 - gate) * X
        return hidden_states_MSA
