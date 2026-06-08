import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

_TAG = "mask"


class Mask(nn.Module):
    """upstream: static adj mask (element-wise multiply)
    改动: Straight-through Gumbel top-k mask
    训练时用Gumbel噪声软采样, 推理时硬top-k
    让模型自适应地选择哪些边重要, 不受预定义adj约束
    """

    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 改动: 可学习的top-k比例
        self.keep_ratio = nn.Parameter(torch.tensor(0.3))
        # Gumbel温度 (训练时退火)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _gumbel_topk(self, adj, k):
        """Straight-through Gumbel top-k: forward用硬mask, backward用软梯度"""
        if self.training:
            # 加Gumbel噪声
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(adj) + 1e-8) + 1e-8)
            perturbed = adj + gumbel_noise * self.temperature.abs().clamp(min=0.1)
        else:
            perturbed = adj

        # top-k selection
        if k < adj.shape[-1]:
            topk_vals, topk_idx = torch.topk(perturbed, k, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk_idx, 1.0)
            # Straight-through: forward用硬mask, backward通过adj传梯度
            result = adj * (mask - adj.detach() + adj)
        else:
            result = adj
        return result

    def _mask(self, index, adj):
        base_mask = self.mask[index] + torch.ones_like(
            self.mask[index]) * 1e-7
        masked = base_mask.to(adj.device) * adj

        # 改动: Gumbel top-k on the masked adjacency
        num_nodes = adj.shape[-1]
        k = max(1, int(num_nodes * self.keep_ratio.sigmoid().item()))
        result = self._gumbel_topk(masked, k)

        _dbg(f"{_TAG}/keep_ratio",
             self.keep_ratio.sigmoid(), _TAG)
        return result

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._mask(index, a))
        return result
