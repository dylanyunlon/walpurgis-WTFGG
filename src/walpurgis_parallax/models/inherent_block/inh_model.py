"""
InhModel — Parallax变体 (M054)
算法改动: xLSTM + Positional Interpolation
  原版: 标准GRUCell + MultiheadSelfAttention
  Parallax:
    - xLSTM: 扩展LSTM, 加入指数门控(exponential gating)
      和记忆混合(memory mixing)机制
      forget gate用指数函数: f = exp(W_f * [h, x])
      这让遗忘门能"更快地遗忘" — 适合长序列
      新增: normalizer state n_t 稳定cell state
    - Positional Interpolation: 位置编码不是固定正弦,
      而是在训练长度和推理长度之间做插值
      pe(pos) = sincos(pos * L_train / L_infer)
      允许外推到更长序列而不退化

  xLSTM参考: Beck et al. "xLSTM: Extended Long Short-Term Memory"
"""
import math
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from ... import _dbg


class XLSTMLayer(nn.Module):
    """xLSTM: 指数门控 + 归一化器状态"""

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 输入门
        self.W_i = nn.Linear(hidden_dim * 2, hidden_dim)
        # 遗忘门 — 指数门控
        self.W_f = nn.Linear(hidden_dim * 2, hidden_dim)
        # 输出门
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim)
        # cell候选
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

        # 记忆混合门: 控制新旧cell的混合比
        self.W_mix = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        # 指数门控的上界截断 — 防止溢出
        self._exp_clip = 10.0

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)

        # 初始化隐状态、cell状态、normalizer
        hx = torch.zeros_like(X[:, 0, :])
        cx = torch.zeros_like(X[:, 0, :])
        nx = torch.ones_like(X[:, 0, :])  # normalizer

        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            combined = torch.cat([hx, x_t], dim=-1)

            # 输入门: sigmoid
            i_t = torch.sigmoid(self.W_i(combined))
            # 遗忘门: 指数门控 — exp(clamp(linear))
            f_raw = self.W_f(combined)
            f_t = torch.exp(torch.clamp(
                f_raw, max=self._exp_clip))
            # 输出门: sigmoid
            o_t = torch.sigmoid(self.W_o(combined))
            # cell候选
            c_tilde = torch.tanh(self.W_c(x_t))

            # xLSTM核心: 指数遗忘 + 归一化
            cx = f_t * cx + i_t * c_tilde
            nx = f_t * nx + i_t  # normalizer跟踪

            # 归一化cell state — 防止指数爆炸
            cx_normed = cx / (nx + 1e-8)

            # 记忆混合: 融合归一化和原始cell
            mix = torch.sigmoid(self.W_mix(combined))
            cx_mixed = mix * cx_normed + (1 - mix) * torch.tanh(cx)

            # 输出
            hx = o_t * torch.tanh(cx_mixed)
            output.append(hx)

        output = torch.stack(output, dim=0)
        output = self.dropout(output)

        _dbg("xlstm.output_norm",
             output.norm(), "inherent")
        _dbg("xlstm.normalizer_range",
             f"[{nx.min().item():.4f}, {nx.max().item():.4f}]",
             "inherent")
        return output


class PositionalInterpolation(nn.Module):
    """位置插值编码 — 支持序列长度外推"""

    def __init__(self, d_model, dropout=None,
                 max_len=5000, train_len=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.train_len = train_len
        # 可学习的缩放因子
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        # 预计算正弦编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        seq_len = X.size(0)
        # 位置插值: 当seq_len > train_len时, 缩放位置索引
        if seq_len > self.train_len:
            ratio = self.train_len / seq_len
            positions = torch.arange(
                seq_len, device=X.device).float() * ratio
            positions = positions.long().clamp(
                max=self.pe.size(0) - 1)
            pe_interp = self.pe[positions]
        else:
            pe_interp = self.pe[:seq_len]

        scale = torch.clamp(self.pe_scale, min=0.01, max=5.0)
        X = X + scale * pe_interp
        X = self.dropout(X)
        return X


class CrossAttentionLayer(nn.Module):
    """跨模态交叉注意力: Q=xLSTM输出, K/V=原始信号"""

    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.cross_attn = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        self.self_attn = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        # 门控: cross vs self
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Q, K, V, cross_K=None, cross_V=None):
        self_out = self.self_attn(Q, K, V)[0]
        self_out = self.dropout(self_out)

        if cross_K is not None and cross_V is not None:
            cross_out = self.cross_attn(
                Q, cross_K, cross_V)[0]
            cross_out = self.dropout(cross_out)
            g = torch.sigmoid(self.gate)
            combined = g * self_out + (1 - g) * cross_out
            _dbg("cross_attn.gate", g, "inherent")
        else:
            combined = self_out

        combined = self.norm(combined)
        return combined
