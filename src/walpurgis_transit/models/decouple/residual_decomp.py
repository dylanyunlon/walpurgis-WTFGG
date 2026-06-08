"""
ResidualDecomp — Transit变体
算法改动: EMD经验模态分解 (Empirical Mode Decomposition)
  原版: LayerNorm + ReLU 直接做残差
  Transit: 将信号分解为近似IMF(Intrinsic Mode Function)分量
           - 用可学习的滤波器组模拟sifting过程
           - 提取趋势/振荡/残差三个层次
           - 通过可学习权重重组分量做残差
           EMD的核心思想: 信号 = Σ(IMF_k) + residue
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg


class LearnableEMDSifter(nn.Module):
    """可学习的EMD筛选器: 用1D卷积模拟包络提取和均值计算"""

    def __init__(self, dim, num_imf=3, kernel_size=3):
        super().__init__()
        self.num_imf = num_imf
        # 每个IMF层级用一组不同感受野的卷积提取
        self.envelope_extractors = nn.ModuleList()
        for k in range(num_imf):
            # 感受野递增: 更深层捕获更低频分量
            ks = kernel_size + 2 * k
            pad = ks // 2
            self.envelope_extractors.append(nn.Sequential(
                nn.Conv1d(dim, dim, ks, padding=pad, groups=dim),
                nn.ELU(inplace=True),
                nn.Conv1d(dim, dim, 1),
            ))
        # IMF分量权重: 控制各分量对残差的贡献
        self.imf_weights = nn.Parameter(
            torch.ones(num_imf + 1) / (num_imf + 1))

    def forward(self, x):
        """x: [*, D], 对最后一维做EMD风格分解
        返回加权残差"""
        orig_shape = x.shape
        # 重塑为 [batch, D, seq] 以适配Conv1d
        if x.dim() == 4:
            B, L, N, D = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        elif x.dim() == 3:
            B, L, D = x.shape
            N = None
            x_flat = x.permute(0, 2, 1)  # [B, D, L]
        else:
            return x

        residual = x_flat
        imf_components = []
        for extractor in self.envelope_extractors:
            # 提取包络均值 (模拟sifting中的上下包络平均)
            envelope_mean = extractor(residual)
            # 当前IMF ≈ 信号 - 包络均值
            imf = residual - envelope_mean
            imf_components.append(imf)
            # 更新残差
            residual = envelope_mean

        # 最后的残差是趋势项
        imf_components.append(residual)

        # 加权重组
        w = F.softmax(self.imf_weights, dim=0)
        result = sum(w[i] * comp for i, comp in enumerate(imf_components))

        # 恢复形状
        if N is not None:
            result = result.reshape(B, N, D, L).permute(0, 3, 1, 2)
        else:
            result = result.permute(0, 2, 1)

        return result


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        self.emd_sifter = LearnableEMDSifter(dim, num_imf=3)
        # 残差分离强度
        self.decomp_strength = nn.Parameter(torch.tensor(0.8))
        # 最终归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        strength = torch.sigmoid(self.decomp_strength)
        # EMD分解: 从y中提取结构化分量
        y_decomposed = self.emd_sifter(y)
        # 残差 = 原始信号 - 强度 * 分解后信号
        u = x - strength * y_decomposed
        u = self.norm(u)

        _dbg("residual_decomp.strength",
             strength, "decouple")
        _dbg("residual_decomp.imf_weights",
             self.emd_sifter.imf_weights, "decouple")

        return u
