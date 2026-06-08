"""
Aphelion Mask — 算法改写 #5:
  upstream: element-wise mask with pre-defined adjacency
  corona: top-k sparse mask (只保留top-k个邻居)
  aphelion: Entropy-regularized mask — 在mask过程中加入信息熵正则化,
            鼓励mask分布接近均匀分布(最大熵原则), 防止图结构过于稀疏
            或过于集中。mask权重通过softmax归一化后加上熵惩罚项。
  改动幅度: ~25% (熵正则化替代简单top-k)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Aphelion: 熵正则化的强度系数 (可学习)
        self.entropy_weight = nn.Parameter(torch.tensor(0.1))

    def _mask(self, index, adj):
        # 安全处理: 如果预定义adj数量不够, 循环使用
        safe_idx = index % len(self.mask)
        mask = self.mask[safe_idx] + torch.ones_like(self.mask[safe_idx]) * 1e-7
        masked = mask.to(adj.device) * adj
        # Aphelion: entropy-regularized mask — softmax归一化 + 熵正则
        if masked.dim() == 3:
            # softmax沿最后一维, 使权重和为1
            soft_mask = F.softmax(masked, dim=-1)
            # 计算每行的信息熵: H = -sum(p * log(p))
            eps = 1e-8
            entropy = -(soft_mask * torch.log(soft_mask + eps)).sum(dim=-1, keepdim=True)
            # 最大熵 = log(N), 熵正则化: 鼓励mask接近均匀分布
            max_entropy = torch.log(torch.tensor(float(masked.shape[-1]), device=masked.device))
            # 熵奖励: entropy越大越好, 用(max_entropy - entropy)作为惩罚
            entropy_bonus = torch.sigmoid(self.entropy_weight) * (entropy / (max_entropy + eps))
            # 将熵奖励加回mask (广播到最后一维)
            regularized_mask = soft_mask * (1.0 + entropy_bonus)
            return regularized_mask
        return masked

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
