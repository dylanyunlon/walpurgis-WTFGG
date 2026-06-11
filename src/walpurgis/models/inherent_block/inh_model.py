"""Cascade inherent model: GRU + LayerNorm + Transformer with residual scaling.
Unlike upstream (standard GRU + vanilla MHA) and vortex (MinGRU + relative positional bias),
Cascade uses standard GRU with post-step LayerNorm for stable recurrence and
a Transformer layer with learnable residual scaling factor to control the
contribution of attention relative to the input."""
import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        # Cascade特有: LayerNorm after each GRU step for gradient stability
        self.ln = nn.LayerNorm(hidden_dim)
        # ═══ 从TITAN移植: 多尺度EMA时序混合 (鲁迅拿法, 改写~20%) ═══
        # 原理: TITAN用mixture-of-experts捕获多时间尺度,我们简化为多α EMA融合
        # 三个EMA分支(fast=0.8/mid=0.5/slow=0.2)并行追踪GRU输出,
        # 可学习gate决定各尺度贡献比 — 短期突变用fast, 长期趋势用slow
        self._ema_alphas = [0.8, 0.5, 0.2]  # fast/mid/slow
        self.ema_gate = nn.Linear(hidden_dim * 3, 3, bias=False)
        self.ema_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 门控初始化: 初期以原始GRU为主, EMA缓慢介入
        self.ema_inject_gate = nn.Parameter(th.tensor(-2.0))  # sigmoid(-2)≈0.12
        # ═══ 频域低通滤波参数 (鲁迅拿法, 从TITAN移植) ═══
        # spectral_gate: 决定低频截止点, 初始0.0 → sigmoid mask在中频处=0.5
        # 训练中学习: 正值→保留更多低频, 负值→允许更多高频
        self.spectral_gate = nn.Parameter(th.tensor(0.0))
        # spectral_inject_gate: sigmoid(-3)≈0.047, 初期几乎不影响
        self.spectral_inject_gate = nn.Parameter(th.tensor(-3.0))

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim]    = X.shape
        X   = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx  = th.zeros_like(X[:, 0, :])
        # 初始化多尺度EMA状态
        ema_states = [th.zeros_like(hx) for _ in self._ema_alphas]
        output  = []
        for t in range(X.shape[1]):
            hx  = self.gru_cell(X[:, t, :], hx)
            # Cascade: LayerNorm after GRU for stable gradients
            hx  = self.ln(hx)
            # ═══ 多尺度EMA混合 ═══
            for i, alpha in enumerate(self._ema_alphas):
                ema_states[i] = alpha * hx + (1 - alpha) * ema_states[i]
            # 门控融合: concat三个尺度 → softmax权重 → 加权和
            ema_cat = th.cat(ema_states, dim=-1)  # [BN, 3*D]
            ema_weights = th.softmax(self.ema_gate(ema_cat), dim=-1)  # [BN, 3]
            ema_mix = sum(ema_weights[:, i:i+1] * ema_states[i] for i in range(3))
            # 门控注入: sigmoid gate控制EMA的影响力
            inject = th.sigmoid(self.ema_inject_gate)
            hx = hx + inject * self.ema_proj(ema_mix - hx)  # 残差注入
            output.append(hx)
        # ═══ 频域低通滤波残差注入 (鲁迅拿法, ~20%改写) ═══
        # 原理: 交通流是周期性信号, FFT后截断高频(噪声), 只保留低频成分
        # 实现: 对时间序列output做FFT, 可学习gate决定低频保留比例
        # 改写点vs原始FFT滤波: 不硬截断而是用sigmoid mask软截断, 门控残差注入
        output_stack = th.stack(output, dim=1)  # [BN, T, D]
        if output_stack.shape[1] >= 4:  # 序列足够长才做频域处理
            fft_out = th.fft.rfft(output_stack, dim=1)  # [BN, T//2+1, D]
            freq_len = fft_out.shape[1]
            # 可学习低频权重: 低频保留, 高频衰减; 初始保留前50%
            freq_idx = th.arange(freq_len, dtype=th.float32,
                                 device=output_stack.device)
            freq_mask = th.sigmoid(
                self.spectral_gate - freq_idx / (freq_len * 0.5))
            freq_mask = freq_mask.unsqueeze(0).unsqueeze(-1)  # [1, F, 1]
            fft_filtered = fft_out * freq_mask
            filtered = th.fft.irfft(
                fft_filtered, n=output_stack.shape[1], dim=1)  # [BN, T, D]
            # 门控残差注入: sigmoid gate控制频域修正的影响
            spec_inject = th.sigmoid(self.spectral_inject_gate)
            output_stack = output_stack + spec_inject * (filtered - output_stack)
            if _CAS_DBG:
                print(f"[CAS:spectral@inh_model] spec_inject={spec_inject.item():.4f} "
                      f"spectral_gate={self.spectral_gate.item():.4f} "
                      f"freq_mask_mean={freq_mask.mean().item():.4f}",
                      file=sys.stderr)
            output = [output_stack[:, t] for t in range(output_stack.shape[1])]
        output  = th.stack(output, dim=0)
        output  = self.dropout(output)
        if _CAS_DBG:
            print(f"[CAS:rnn@inh_model] hidden_mean={hx.mean().item():.4f} "
                  f"std={hx.std().item():.4f} "
                  f"ema_gate={inject.item():.4f} "
                  f"ema_w=[{ema_weights[:, 0].mean().item():.3f}/"
                  f"{ema_weights[:, 1].mean().item():.3f}/"
                  f"{ema_weights[:, 2].mean().item():.3f}]", file=sys.stderr)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention  = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout                    = nn.Dropout(dropout)
        # Cascade特有: learnable residual scaling factor
        # Controls how much attention output contributes vs input
        self.res_scale = nn.Parameter(th.tensor(0.5))

    def forward(self, X, K, V):
        hidden_states_MSA   = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA   = self.dropout(hidden_states_MSA)
        # Cascade: scaled residual — alpha * attention + (1-alpha) * input
        alpha = th.sigmoid(self.res_scale)
        hidden_states_MSA = alpha * hidden_states_MSA + (1.0 - alpha) * X
        if _CAS_DBG:
            print(f"[CAS:transformer@inh_model] res_alpha={alpha.item():.4f} "
                  f"attn_norm={hidden_states_MSA.detach().norm().item():.4f}",
                  file=sys.stderr)
        return hidden_states_MSA
