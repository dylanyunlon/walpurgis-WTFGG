"""
InhModel — Transit变体
算法改动: S4 Structured State Space + ELU
  原版: 标准GRUCell + MultiheadSelfAttention
  Transit:
    - S4层: 结构化状态空间模型, 通过离散化连续SSM实现
      x'(t) = Ax(t) + Bu(t)
      y(t) = Cx(t) + Du(t)
      离散化: x_k = Ā*x_{k-1} + B̄*u_k, y_k = C*x_k + D*u_k
      用HiPPO矩阵初始化A实现长距离依赖
    - ELU激活替代tanh: 正半轴线性, 负半轴指数衰减
    - 自注意力保持但用Gated-Attention: 输出经过门控
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from ... import _dbg


class S4Layer(nn.Module):
    """Simplified S4 (Structured State Space Sequence model)
    离散化的状态空间: x_k = A_bar * x_{k-1} + B_bar * u_k
                     y_k = C * x_k + D * u_k
    用对角化近似让训练高效
    """

    def __init__(self, d_model, state_dim=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        # 用对角化A矩阵: 每个维度独立的衰减率
        # 初始化用HiPPO-LegS思想: 衰减率按 -1/2 + n 分布
        init_A_real = -0.5 * torch.ones(d_model, state_dim)
        init_A_imag = math.pi * torch.arange(state_dim).float().unsqueeze(0).expand(d_model, -1)
        # 存储为log形式以保证稳定性
        self.log_A_real = nn.Parameter(
            torch.log(-init_A_real + 1e-4))
        self.A_imag = nn.Parameter(init_A_imag)
        # B, C: 输入输出映射
        self.B = nn.Parameter(
            torch.randn(d_model, state_dim) * 0.01)
        self.C = nn.Parameter(
            torch.randn(d_model, state_dim) * 0.01)
        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        # 离散化步长 Δ
        self.log_dt = nn.Parameter(
            torch.rand(d_model) * 0.1 - 2.0)
        self.dropout = nn.Dropout(dropout)
        # 输出投影 + ELU
        self.out_proj = nn.Linear(d_model, d_model)

    def _discretize(self):
        """ZOH离散化: Ā = exp(A*Δ), B̄ = (Ā-I)*A^{-1}*B ≈ Δ*B"""
        dt = F.softplus(self.log_dt)  # [d_model]
        A_real = -F.softplus(self.log_A_real)  # 确保负实部
        # 对角A的指数: exp((a+bi)*dt) = exp(a*dt) * (cos(b*dt) + i*sin(b*dt))
        # 简化为实数版: 只用实部衰减
        A_bar = torch.exp(A_real * dt.unsqueeze(-1))  # [d_model, state_dim]
        B_bar = dt.unsqueeze(-1) * self.B  # [d_model, state_dim]
        return A_bar, B_bar

    def forward(self, u):
        """u: [seq_len, batch, d_model] → y: [seq_len, batch, d_model]"""
        seq_len, batch, d_model = u.shape
        A_bar, B_bar = self._discretize()

        # 逐步递推 (对短序列足够快)
        x = torch.zeros(batch, d_model, self.state_dim,
                         device=u.device)
        ys = []
        for t in range(seq_len):
            u_t = u[t]  # [batch, d_model]
            # x_k = A_bar * x_{k-1} + B_bar * u_k
            x = A_bar.unsqueeze(0) * x + B_bar.unsqueeze(0) * u_t.unsqueeze(-1)
            # y_k = C * x_k + D * u_k
            y_t = (self.C.unsqueeze(0) * x).sum(-1) + self.D * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=0)  # [seq_len, batch, d_model]
        y = F.elu(y, inplace=False)  # ELU激活
        y = self.out_proj(y)
        y = self.dropout(y)

        _dbg("s4.output_norm", y.norm(), "inherent")
        _dbg("s4.A_decay_range",
             f"[{A_bar.min():.4f}, {A_bar.max():.4f}]", "inherent")
        return y


class GatedSelfAttention(nn.Module):
    """门控自注意力: 输出经过sigmoid门控"""

    def __init__(self, hidden_dim, num_heads=4,
                 dropout=0.1, bias=True):
        super().__init__()
        self.self_attn = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        # 门控: 控制attention输出的通过比例
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Q, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        attn_out = self.self_attn(Q, K, V)[0]
        attn_out = self.dropout(attn_out)

        # 门控: concat(Q, attn_out) → sigmoid → element-wise gate
        gate_input = torch.cat([Q, attn_out], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        gated = gate * attn_out + (1 - gate) * Q  # 残差门控

        _dbg("gated_attn.gate_mean",
             gate.mean(), "inherent")

        gated = self.norm(gated)
        return gated
