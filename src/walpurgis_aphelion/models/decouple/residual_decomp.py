"""
Aphelion ResidualDecomp — 算法改写 #2:
  upstream: LayerNorm(x - ReLU(y))
  corona: EMA-based decomposition — 指数移动平均平滑残差
  aphelion: VMD (Variational Mode Decomposition) 风格分解 —
            将信号分解为K个可学习频带的模态, 取低频模态作为趋势,
            高频模态作为残差。每个模态有可学习的中心频率和带宽。
  改动幅度: ~30% (多模态频率分解替代简单减法/EMA)
"""
import torch
import torch.nn as nn
from ... import _dbg, VMDStateMonitor

_vmd_monitor = VMDStateMonitor()


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        hidden_dim = input_shape[-1]
        self.ln = nn.LayerNorm(hidden_dim)
        # Aphelion改写: VMD风格分解 — K个可学习的频带滤波器
        # 每个模态有中心频率(omega)和带宽(alpha)
        self.num_modes = 3  # 分解为3个模态: 低频趋势 + 中频 + 高频残差
        # 可学习的中心频率 (初始化为均匀分布在[0,π])
        self.omega = nn.Parameter(torch.linspace(0.1, 2.5, self.num_modes))
        # 可学习的带宽控制 (alpha越大带宽越窄)
        self.alpha = nn.Parameter(torch.ones(self.num_modes) * 2.0)
        # 模态融合权重: 决定哪些模态归入"趋势"
        self.mode_weights = nn.Parameter(torch.tensor([0.7, 0.2, 0.1]))
        # 残差投影
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y):
        # Aphelion: VMD风格分解 — 在特征维度上用可学习的频带滤波
        # y是预测的backcast, 用VMD分解来提取趋势和残差
        B, S, N, D = y.shape

        # 对特征维度构造频率域表示 (简化的VMD: 在特征空间做频带滤波)
        # 用可学习的高斯频带滤波器
        freq_axis = torch.linspace(0, 3.14, D, device=y.device)  # [D]
        modes = []
        for k in range(self.num_modes):
            # 高斯带通滤波器: exp(-alpha_k * (f - omega_k)^2)
            center = torch.sigmoid(self.omega[k]) * 3.14  # 限制在[0, π]
            bandwidth = F.softplus(self.alpha[k])  # 正的带宽
            filter_k = torch.exp(-bandwidth * (freq_axis - center) ** 2)  # [D]
            # 应用滤波器到y的特征维度
            mode_k = y * filter_k.view(1, 1, 1, D)  # [B, S, N, D]
            modes.append(mode_k)

        # 加权重建趋势信号
        weights = torch.softmax(self.mode_weights, dim=0)
        trend = sum(w * m for w, m in zip(weights, modes))

        # 记录VMD状态用于诊断
        mode_energies = [m.detach().norm().item() for m in modes]
        residual = x - trend
        _vmd_monitor.record("residual_decomp", mode_energies, residual.detach())

        # 残差 = 输入 - 趋势, 再投影 + LayerNorm
        u = self.residual_proj(residual)
        u = self.ln(u)
        return u


# 需要F用于softplus
import torch.nn.functional as F
