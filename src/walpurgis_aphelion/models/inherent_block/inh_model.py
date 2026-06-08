"""
Aphelion InhModel — 算法改写 #7:
  upstream: GRUCell + MultiheadAttention
  corona: LSTMCell + RoPE (旋转位置编码)
  aphelion: Retention network + cross-scale fusion —
            用RetNet的多尺度指数衰减机制替代GRU/LSTM,
            不同head用不同的衰减率(gamma)实现多尺度时序建模,
            然后通过cross-scale fusion将不同尺度的特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import retention_state_dump


class RetentionLayer(nn.Module):
    """Aphelion: Retention机制 — 基于指数衰减的递归序列建模
    与Transformer的softmax注意力不同, Retention用固定的指数衰减权重,
    支持递归和并行两种计算模式, 这里用递归模式
    """
    def __init__(self, hidden_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # 每个head的Q/K/V投影
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        # 每个head有不同的衰减率gamma (可学习, sigmoid映射到(0,1))
        # 多尺度: 不同head关注不同时间跨度
        gamma_init = torch.linspace(0.8, 0.999, n_heads)
        self.gamma_logit = nn.Parameter(torch.log(gamma_init / (1 - gamma_init)))

        # GroupNorm for stability
        self.group_norm = nn.GroupNorm(n_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        X: [batch*nodes, seq_len, hidden_dim]
        返回: [seq_len, batch*nodes, hidden_dim]  (与corona的RNNLayer输出格式一致)
        """
        B, T, D = X.shape
        # Q, K, V投影
        Q = self.W_Q(X).view(B, T, self.n_heads, self.head_dim)
        K = self.W_K(X).view(B, T, self.n_heads, self.head_dim)
        V = self.W_V(X).view(B, T, self.n_heads, self.head_dim)

        # 获取各head的衰减率
        gammas = torch.sigmoid(self.gamma_logit)  # [n_heads]

        # 递归Retention: S_t = gamma * S_{t-1} + K_t^T V_t
        # output_t = Q_t S_t
        outputs = []
        # 初始化递归状态: [B, n_heads, head_dim, head_dim]
        S = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim,
                        device=X.device, dtype=X.dtype)

        for t in range(T):
            # K_t: [B, n_heads, head_dim] → [B, n_heads, head_dim, 1]
            K_t = K[:, t, :, :].unsqueeze(-1)
            # V_t: [B, n_heads, head_dim] → [B, n_heads, 1, head_dim]
            V_t = V[:, t, :, :].unsqueeze(-2)

            # 递归更新: S_t = gamma * S_{t-1} + K_t^T V_t
            # gamma广播: [n_heads] → [1, n_heads, 1, 1]
            gamma_t = gammas.view(1, self.n_heads, 1, 1)
            S = gamma_t * S + torch.matmul(K_t, V_t)

            # 输出: Q_t S_t → [B, n_heads, head_dim]
            Q_t = Q[:, t, :, :]  # [B, n_heads, head_dim]
            # [B, n_heads, head_dim] @ [B, n_heads, head_dim, head_dim]
            out_t = torch.einsum('bnh,bnhd->bnd', Q_t, S)
            outputs.append(out_t)

        # 诊断: 打印最终递归状态和衰减因子
        retention_state_dump("retention_final", S.detach().flatten(0, 1).mean(0), gammas.detach())

        # 堆叠: [T, B, n_heads, head_dim]
        output = torch.stack(outputs, dim=1)  # [B, T, n_heads, head_dim]
        output = output.reshape(B, T, self.hidden_dim)

        # GroupNorm + 输出投影
        output = output.transpose(1, 2)  # [B, D, T] for GroupNorm
        output = self.group_norm(output)
        output = output.transpose(1, 2)  # [B, T, D]
        output = self.W_O(output)
        output = self.dropout(output)

        # 转换为 [T, B, D] 格式 (与corona RNNLayer输出一致)
        output = output.transpose(0, 1)
        return output


class CrossScaleFusion(nn.Module):
    """Aphelion: 跨尺度融合模块 — 融合不同retention head(不同gamma)的多尺度特征"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, fine_scale, coarse_scale):
        """fine_scale: 快衰减head的输出(局部), coarse_scale: 慢衰减head的输出(全局)"""
        combined = torch.cat([fine_scale, coarse_scale], dim=-1)
        gate = self.gate(combined)
        fused = gate * self.transform(fine_scale) + (1 - gate) * coarse_scale
        return fused


class RNNLayer(nn.Module):
    """Aphelion: Retention替代GRU/LSTM"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Aphelion改写: Retention网络替代LSTM
        self.retention = RetentionLayer(hidden_dim, n_heads=2, dropout=dropout or 0.1)
        # cross-scale fusion
        self.cross_scale = CrossScaleFusion(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        # 保留一个简单的线性层用于forecast中的递归预测
        self._forecast_cell = nn.Linear(hidden_dim, hidden_dim)
        self._last_hidden = None

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)

        # Aphelion: Retention前向 → [T, B*N, D]
        output = self.retention(X)

        # cross-scale fusion: 将前半段和后半段特征做融合
        T = output.shape[0]
        if T > 1:
            mid = T // 2
            fine = output[mid:, :, :]    # 后半段(更近的时步)
            coarse = output[:mid, :, :]  # 前半段(更远的时步)
            # 对齐长度
            min_t = min(fine.shape[0], coarse.shape[0])
            fine_aligned = fine[:min_t]
            coarse_aligned = coarse[:min_t]
            fused = self.cross_scale(fine_aligned, coarse_aligned)
            # 拼接: 非融合部分 + 融合部分
            output = torch.cat([output[:T - min_t], fused], dim=0)

        # 保存最后一步hidden用于forecast递归
        self._last_hidden = output[-1].detach()

        output = self.dropout_layer(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.multi_head_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
