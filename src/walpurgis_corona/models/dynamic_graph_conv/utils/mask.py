"""
Corona Mask — top-k sparse mask
"""
import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        self.top_k = min(5, model_args.get('num_nodes', 10))

    def _mask(self, index, adj):
        # 安全处理: 如果预定义adj数量不够, 循环使用
        safe_idx = index % len(self.mask)
        mask = self.mask[safe_idx] + torch.ones_like(self.mask[safe_idx]) * 1e-7
        masked = mask.to(adj.device) * adj
        # Corona: top-k稀疏化
        if masked.dim() == 3:
            k = min(self.top_k, masked.shape[-1])
            topk_vals, topk_idx = torch.topk(masked, k=k, dim=-1)
            sparse_mask = torch.zeros_like(masked)
            sparse_mask.scatter_(-1, topk_idx, topk_vals)
            return sparse_mask
        return masked

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
