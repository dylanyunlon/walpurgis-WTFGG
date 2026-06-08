import torch
import torch.nn as nn

class Mask(nn.Module):
    """Helix改写: 使用可学习的mask阈值替代固定epsilon,
    允许网络自适应调整mask的松紧度"""
    def __init__(self, **model_args):
        super().__init__()
        self.mask   = model_args['adjs']
        # Helix特有: 可学习的mask epsilon
        self.mask_eps = nn.Parameter(torch.tensor(-7.0))  # sigmoid(-7) ≈ 1e-3

    def _mask(self, index, adj):
        eps = torch.sigmoid(self.mask_eps) * 0.01  # 范围约(0, 0.01)
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * eps
        return mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
