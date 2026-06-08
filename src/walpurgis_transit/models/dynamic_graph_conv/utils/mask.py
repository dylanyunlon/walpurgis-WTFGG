"""
Mask — Transit变体
算法改动: Differentiable Binary Mask (二值掩码 + straight-through + L0正则化)
  原版: 直接用predefined adj做元素乘法
  Transit:
    - 为每条边学习一个连续参数 log_alpha
    - 训练时: 用Hard Concrete分布采样近似二值掩码
      z = sigmoid((log(u/(1-u)) + log_alpha) / β)
      z̄ = z * (ζ - γ) + γ, 然后clamp到[0,1]
    - 推理时: 用sigmoid(log_alpha)做确定性掩码
    - L0正则化: 鼓励稀疏, penalty = Σ sigmoid(log_alpha - β*log(-γ/ζ))
"""
import torch
import torch.nn as nn
from .... import _dbg


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Hard Concrete参数
        self.beta = 0.66  # 温度
        self.gamma = -0.1  # 左拉伸
        self.zeta = 1.1   # 右拉伸
        # 为每条边学习 log_alpha (初始化由predefined adj决定)
        self.log_alphas = nn.ParameterList()
        for adj_tensor in self.mask:
            # 用adj的值初始化: 高连接性 → 大log_alpha
            init = torch.log(adj_tensor.clamp(min=1e-4) /
                             (1 - adj_tensor.clamp(max=1-1e-4) + 1e-8))
            self.log_alphas.append(nn.Parameter(init.clone()))

    def _hard_concrete_sample(self, log_alpha):
        """Hard Concrete采样: 可微的二值掩码"""
        if self.training:
            u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
            # Concrete relaxation
            s = torch.sigmoid(
                (torch.log(u / (1 - u)) + log_alpha) / self.beta)
            # 拉伸到 [gamma, zeta]
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            # Clamp到[0,1]: 实现硬边界
            z = s_bar.clamp(0.0, 1.0)
        else:
            # 推理时: 确定性
            z = torch.sigmoid(log_alpha)
            z = (z * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)
        return z

    def l0_penalty(self):
        """L0正则化: 估计非零掩码元素数量的期望
        用于在总loss里加正则"""
        total = 0.0
        for la in self.log_alphas:
            # P(z > 0) = sigmoid(log_alpha - beta * log(-gamma/zeta))
            prob_nonzero = torch.sigmoid(
                la - self.beta * torch.log(
                    torch.tensor(-self.gamma / self.zeta)))
            total = total + prob_nonzero.sum()
        return total

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            z = self._hard_concrete_sample(self.log_alphas[index])
            masked = z.to(a.device) * a
            result.append(masked)

        _dbg("mask.l0_penalty",
             f"{self.l0_penalty().item():.4f}", "graph")
        _dbg("mask.sparsity_0",
             f"{(result[0] < 0.01).float().mean():.3f}", "graph")

        return result
