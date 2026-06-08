"""
InhModel — Umbra变体
算法改动: Mamba SSM + ALiBi位置偏差
  原版: 标准GRUCell + MultiheadSelfAttention
  Umbra:
    - Mamba-style SSM: 选择性状态空间模型
      核心: 输入依赖的 B(x), C(x) 矩阵 + 离散化
      h_t = A_bar * h_{t-1} + B_bar * x_t
      y_t = C * h_t
      其中A_bar, B_bar通过ZOH离散化从连续参数得到
      选择性机制: Δ(delta)参数由输入x决定, 控制信息保留/遗忘
    - ALiBi (Attention with Linear Biases): 位置偏差
      不用正弦位置编码, 直接给attention score加线性距离惩罚
      score_ij -= m * |i - j|, m为每个head的固定斜率
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from ... import _dbg


class MambaSSMLayer(nn.Module):
    """简化版Mamba选择性状态空间模型"""

    def __init__(self, hidden_dim, state_dim=None, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim or max(hidden_dim // 2, 4)

        # 输入投影
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        # 选择性参数: 从输入x产出Δ, B, C
        self.delta_proj = nn.Linear(hidden_dim, self.state_dim)
        self.B_proj = nn.Linear(hidden_dim, self.state_dim)
        self.C_proj = nn.Linear(hidden_dim, self.state_dim)
        # 连续A参数 (固定为负数确保稳定)
        A_log = torch.log(torch.arange(1, self.state_dim + 1).float())
        self.A_log = nn.Parameter(A_log)
        # 输出投影
        self.D = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 用于forecast步进的辅助参数
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X_flat = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)

        # 输入门控: 分成两路, 一路走SSM, 一路做门控
        xz = self.in_proj(X_flat)
        x_ssm, z = xz.chunk(2, dim=-1)
        z = F.silu(z)

        # 选择性参数
        delta = F.softplus(self.delta_proj(x_ssm))  # [B*N, L, state_dim], 正数
        B = self.B_proj(x_ssm)                       # [B*N, L, state_dim]
        C = self.C_proj(x_ssm)                       # [B*N, L, state_dim]

        # 连续A → 离散A_bar, B_bar (ZOH离散化)
        A = -torch.exp(self.A_log)  # [state_dim], 负数保稳定
        # A_bar = exp(delta * A): [B*N, L, state_dim]
        A_bar = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
        # B_bar = delta * B
        B_bar = delta * B

        # SSM扫描 (sequential scan, 无法并行但序列短可接受)
        # 将x_ssm从hidden_dim投影到state_dim用于SSM输入
        BN, L, S = B_bar.shape
        x_ssm_proj = x_ssm[..., :S] if hidden_dim >= S else F.pad(x_ssm, (0, S - hidden_dim))
        h = torch.zeros(BN, S, device=X.device)
        outputs = []
        _dbg("mamba_ssm.scan_dims",
             f"BN={BN}, L={L}, S={S}, D={hidden_dim}", "inherent")
        for t in range(L):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * x_ssm_proj[:, t, :]
            # y_t = C_t * h_t → [BN, S], 保留state_dim以便拼接后还原
            y_t = C[:, t, :] * h  # [BN, S] element-wise, 不sum
            outputs.append(y_t.unsqueeze(1))  # [BN, 1, S]

        ssm_out = torch.cat(outputs, dim=1)  # [BN, L, S]
        _dbg("mamba_ssm.ssm_out_shape", ssm_out.shape, "inherent")
        # 从state_dim映射回hidden_dim
        if S < hidden_dim:
            ssm_out = F.pad(ssm_out, (0, hidden_dim - S))
        ssm_out = ssm_out[..., :hidden_dim]  # [BN, L, hidden_dim]
        # 残差D直连: D * x_ssm (skip connection)
        ssm_out = ssm_out + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        # 门控
        y = ssm_out * z
        y = self.out_proj(y)
        y = self.dropout(y)

        # 转成[L, B*N, D]格式以兼容后续transformer接口
        output = y.transpose(0, 1)

        _dbg("mamba_ssm.output_norm", output.norm(), "inherent")
        _dbg("mamba_ssm.A_log_range",
             f"[{self.A_log.min().item():.3f}, {self.A_log.max().item():.3f}]",
             "inherent")
        return output


class ALiBiAttentionLayer(nn.Module):
    """ALiBi注意力: 线性位置偏差代替正弦编码"""

    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # ALiBi斜率: 每个head一个固定斜率 2^{-8/num_heads * (i+1)}
        slopes = torch.tensor([
            2.0 ** (-(8.0 / num_heads) * (i + 1))
            for i in range(num_heads)
        ])
        self.register_buffer('alibi_slopes', slopes)

    def _get_alibi_bias(self, len_q, len_k, device):
        """构造ALiBi位置偏差矩阵 [num_heads, L_q, L_k]
        bias[h, i, j] = -slopes[h] * |i_pos - j_pos|
        当L_q < L_k时(forecast情况), query在序列末尾"""
        pos_k = torch.arange(len_k, device=device).float()
        # query位置对齐到key序列末尾
        pos_q = torch.arange(len_k - len_q, len_k, device=device).float()
        rel_dist = (pos_q.unsqueeze(1) - pos_k.unsqueeze(0)).abs()  # [L_q, L_k]
        # [H, L_q, L_k]
        alibi = -rel_dist.unsqueeze(0) * self.alibi_slopes.view(-1, 1, 1)
        return alibi

    def forward(self, Q, K, V, cross_K=None, cross_V=None):
        L, BN, D = Q.shape
        L_k = K.shape[0]
        H = self.num_heads
        head_d = self.head_dim

        # 自注意力
        q = self.q_proj(Q).view(L, BN, H, head_d).permute(1, 2, 0, 3)
        k = self.k_proj(K).view(L_k, BN, H, head_d).permute(1, 2, 0, 3)
        v = self.v_proj(V).view(V.shape[0], BN, H, head_d).permute(1, 2, 0, 3)

        # [BN, H, L_q, L_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_d)
        # 添加ALiBi偏差: 处理L_q != L_k的情况
        alibi = self._get_alibi_bias(L, L_k, Q.device)  # [H, L_q, L_k]
        scores = scores + alibi.unsqueeze(0)  # broadcast [1,H,Lq,Lk] → [BN,H,Lq,Lk]

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self_out = torch.matmul(attn, v)  # [BN, H, L_q, head_d]
        self_out = self_out.permute(2, 0, 1, 3).reshape(L, BN, D)

        if cross_K is not None and cross_V is not None:
            # 交叉注意力 (也用ALiBi)
            L_ck = cross_K.shape[0]
            ck = self.k_proj(cross_K).view(L_ck, BN, H, head_d).permute(1, 2, 0, 3)
            cv = self.v_proj(cross_V).view(cross_V.shape[0], BN, H, head_d).permute(1, 2, 0, 3)
            cross_scores = torch.matmul(q, ck.transpose(-1, -2)) / math.sqrt(head_d)
            cross_alibi = self._get_alibi_bias(L, L_ck, Q.device)  # [H, L_q, L_ck]
            cross_scores = cross_scores + cross_alibi.unsqueeze(0)
            cross_attn = F.softmax(cross_scores, dim=-1)
            cross_attn = self.dropout(cross_attn)
            cross_out = torch.matmul(cross_attn, cv)
            cross_out = cross_out.permute(2, 0, 1, 3).reshape(L, BN, D)
            # 平均混合
            combined = 0.5 * self_out + 0.5 * cross_out
            _dbg("alibi_attn.cross_enabled", "True", "inherent")
        else:
            combined = self_out

        combined = self.norm(combined)
        return combined
