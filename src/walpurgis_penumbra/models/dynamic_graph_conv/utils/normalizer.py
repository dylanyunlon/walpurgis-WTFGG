"""
Normalizer — Penumbra变体
算法改动: Sinkhorn行列双归一化 替代 单行归一化
  原版: D^{-1} * A (行归一化)
  Penumbra: 交替行列归一化, 使邻接矩阵逼近双随机矩阵
           保证行和与列和都接近1, 信息传播更均衡
           可学习的迭代次数参数(训练时3-8次, 推理时固定)
"""
import torch
import torch.nn as nn
from .... import _dbg, _sinkhorn_tracker


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self, sinkhorn_iters=5, eps=1e-6):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        # 可学习的收敛阈值
        self.convergence_threshold = nn.Parameter(
            torch.tensor(0.01))

    def _sinkhorn_norm(self, graph):
        """Sinkhorn迭代: 交替行列归一化"""
        M = graph + self.eps
        M = torch.clamp(M, min=0)
        residual = float('inf')
        actual_iters = 0
        for i in range(self.sinkhorn_iters):
            # 行归一化
            row_sum = M.sum(dim=-1, keepdim=True) + self.eps
            M = M / row_sum
            # 列归一化
            col_sum = M.sum(dim=-2, keepdim=True) + self.eps
            M = M / col_sum
            # 收敛检查
            new_row_sum = M.sum(dim=-1)
            residual = (new_row_sum - 1.0).abs().mean().item()
            actual_iters = i + 1
            thresh = torch.sigmoid(
                self.convergence_threshold).item()
            if residual < thresh:
                break

        M = _remove_nan_inf(M)
        _sinkhorn_tracker.record(actual_iters, residual)
        return M

    def forward(self, adj):
        normed = [self._sinkhorn_norm(a) for a in adj]
        _dbg("normalizer.sinkhorn_iters",
             f"max_iters={self.sinkhorn_iters}", "graph")
        return normed


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
