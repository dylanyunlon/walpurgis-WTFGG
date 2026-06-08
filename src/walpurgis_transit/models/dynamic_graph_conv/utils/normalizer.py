"""
Normalizer — Transit变体
算法改动: Power-Mean归一化 (幂均值, 可学习幂指数)
  原版: D^{-1} * A (行归一化)
  Transit: 使用广义幂均值 M_p(x) = (mean(x^p))^(1/p)
           当p=1时退化为算术均值(标准行归一化)
           当p→-∞时近似min, p→+∞时近似max
           学习最优的p让图结构自适应选择归一化方式
           + 行列对称化后处理
"""
import torch
import torch.nn as nn
from .... import _dbg


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习幂指数p: 控制归一化行为
        self.power_p = nn.Parameter(torch.tensor(1.0))
        # 行列对称化强度
        self.symmetrize_alpha = nn.Parameter(torch.tensor(0.5))

    def _power_mean_norm(self, graph):
        """Power-Mean行归一化:
        norm_factor_i = (mean_j(a_ij^p))^(1/p)
        a_ij_normed = a_ij / norm_factor_i
        """
        M = graph.clamp(min=0) + self.eps
        p = torch.clamp(self.power_p, min=-3.0, max=3.0)

        # 处理p接近0的情况 (几何均值)
        if p.abs() < 0.05:
            # 几何均值: exp(mean(log(x)))
            log_M = torch.log(M)
            log_mean = log_M.mean(dim=-1, keepdim=True)
            norm_factor = torch.exp(log_mean)
        else:
            # 一般幂均值: (mean(x^p))^(1/p)
            M_p = M.pow(p)
            mean_p = M_p.mean(dim=-1, keepdim=True)
            norm_factor = mean_p.pow(1.0 / p)

        normed = M / (norm_factor + self.eps)

        # 可选: 部分对称化 (行归一化结果 + 转置的加权平均)
        sym_alpha = torch.sigmoid(self.symmetrize_alpha)
        if normed.dim() == 3:
            normed_T = normed.transpose(-1, -2)
        else:
            normed_T = normed.T
        normed = sym_alpha * normed + (1 - sym_alpha) * normed_T

        normed = _remove_nan_inf(normed)
        return normed

    def forward(self, adj):
        normed = [self._power_mean_norm(a) for a in adj]
        _dbg("normalizer.power_p",
             self.power_p, "graph")
        _dbg("normalizer.sym_alpha",
             torch.sigmoid(self.symmetrize_alpha), "graph")
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
