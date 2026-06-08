"""
Normalizer — Umbra变体
算法改动: PageRank-style归一化 (PersonalizedPageRank, 带damping)
  原版: D^{-1} * A (行归一化)
  Umbra: PPR = alpha * (I - (1-alpha) * D^{-1} * A)^{-1}
        用幂迭代近似: pi_{k+1} = (1-alpha) * A_norm * pi_k + alpha * e
        其中alpha=damping factor (可学习), e=uniform或个性化向量
        收敛后的pi就是PPR向量, 每个节点获得全局重要性权重
        信息从"重要"节点流向"不重要"节点时被衰减
"""
import torch
import torch.nn as nn
from .... import _dbg, _pagerank_tracker


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self, damping=0.85, ppr_iters=5, eps=1e-6):
        super().__init__()
        self.ppr_iters = ppr_iters
        self.eps = eps
        # 可学习的damping factor (通常0.85)
        self.damping_logit = nn.Parameter(torch.tensor(1.73))  # sigmoid(1.73) ≈ 0.85
        # 收敛阈值
        self.convergence_threshold = nn.Parameter(torch.tensor(0.01))

    def _ppr_normalize(self, graph):
        """PersonalizedPageRank幂迭代归一化"""
        alpha = torch.sigmoid(self.damping_logit)  # damping ∈ (0,1)
        M = graph.clamp(min=0) + self.eps

        # 行归一化得到转移矩阵
        row_sum = M.sum(dim=-1, keepdim=True) + self.eps
        A_norm = M / row_sum

        # 个性化向量: 均匀分布
        N = M.shape[-1]
        e = torch.ones_like(M[..., :1]) / N if M.dim() == 2 \
            else torch.ones(*M.shape[:-1], 1, device=M.device) / N

        # 幂迭代: pi = (1-alpha) * A_norm^T * pi + alpha * e
        pi = e.expand_as(M[..., :1]).clone()
        if M.dim() == 3:
            # batched: [B, N, N]
            pi = torch.ones(*M.shape[:-1], 1, device=M.device) / N
        else:
            pi = torch.ones(N, 1, device=M.device) / N

        residual = float('inf')
        actual_iters = 0
        for i in range(self.ppr_iters):
            pi_old = pi.clone()
            # pi = (1-alpha)*A_norm^T * pi + alpha * e
            if M.dim() == 3:
                pi = (1.0 - alpha) * torch.bmm(A_norm.transpose(-1, -2), pi) + alpha / N
            else:
                pi = (1.0 - alpha) * torch.mm(A_norm.T, pi) + alpha / N
            residual = (pi - pi_old).abs().mean().item()
            actual_iters = i + 1
            thresh = torch.sigmoid(self.convergence_threshold).item()
            if residual < thresh:
                break

        # 用PPR向量对原图加权: 重要节点连接权重增大
        # pi: [B, N, 1] → [B, N, N] 通过外积
        pi_sq = pi.squeeze(-1)
        # 对称加权: sqrt(pi_i) * A_ij * sqrt(pi_j)
        if M.dim() == 3:
            pi_sqrt = pi_sq.unsqueeze(-1).sqrt()  # [B, N, 1]
            weighted = A_norm * pi_sqrt * pi_sqrt.transpose(-1, -2)
        else:
            pi_sqrt = pi_sq.unsqueeze(-1).sqrt()
            weighted = A_norm * pi_sqrt * pi_sqrt.T

        weighted = _remove_nan_inf(weighted)
        _pagerank_tracker.record(actual_iters, residual)
        return weighted

    def forward(self, adj):
        normed = [self._ppr_normalize(a) for a in adj]
        damping = torch.sigmoid(self.damping_logit)
        _dbg("normalizer.ppr_damping",
             f"alpha={damping.item():.4f}", "graph")
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
