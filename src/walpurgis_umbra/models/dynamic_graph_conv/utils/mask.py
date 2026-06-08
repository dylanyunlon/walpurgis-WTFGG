"""
Mask — Umbra变体
算法改动: Differentiable Top-K (Gumbel-Sinkhorn) 替代 确定性邻接掩码
  原版: 直接用predefined adj做元素乘法
  Umbra: 对邻接矩阵应用可微分的top-k选择
        Gumbel-Sinkhorn: 给log-scores加Gumbel噪声后用Sinkhorn归一化
        逼近排列矩阵, 只保留每行top-k个最大的连接
        k由可学习的sparsity参数控制
        训练时用soft relaxation, 推理时hard top-k
"""
import torch
import torch.nn as nn
from .... import _dbg


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # top-k中的k通过可学习参数控制
        num_nodes = self.mask[0].shape[0]
        # sparsity_logit经过sigmoid后乘以num_nodes得到有效k
        self.sparsity_logit = nn.Parameter(torch.tensor(0.0))
        # Gumbel-Sinkhorn温度
        self.gs_temperature = nn.Parameter(torch.tensor(0.5))
        # Sinkhorn内部迭代次数
        self.sinkhorn_iters = 3

    def _gumbel_sinkhorn_topk(self, scores, base_mask):
        """Gumbel-Sinkhorn可微分Top-K
        scores: 原始邻接权重
        base_mask: predefined adj (用于约束连接范围)
        """
        temp = torch.clamp(self.gs_temperature, min=0.1, max=2.0)
        num_nodes = scores.shape[-1]

        # 有效k: sigmoid(sparsity_logit) * num_nodes, 最少保留2个
        effective_k = torch.sigmoid(self.sparsity_logit) * num_nodes
        effective_k = torch.clamp(effective_k, min=2.0, max=float(num_nodes))

        if self.training:
            # 添加Gumbel噪声
            uniform = torch.rand_like(scores).clamp(1e-7, 1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_scores = (scores + gumbel * 0.1) / temp

            # 使用sigmoid近似代替完整Sinkhorn排列 (更稳定)
            # 思路: 对每行, 把得分转成"被选中概率"
            # 阈值 = 第(N-k)小的值, 用soft threshold
            sorted_vals, _ = noisy_scores.sort(dim=-1, descending=True)
            # k_idx为effective_k的soft索引
            k_float = effective_k
            k_lo = k_float.long().clamp(0, num_nodes - 1)
            k_hi = (k_lo + 1).clamp(0, num_nodes - 1)
            frac = k_float - k_lo.float()
            # 双线性插值得到soft阈值
            thresh = ((1.0 - frac) * sorted_vals[..., k_lo]
                      + frac * sorted_vals[..., k_hi])
            # soft top-k: sigmoid((score - threshold) / temp)
            soft_mask = torch.sigmoid(
                (noisy_scores - thresh.unsqueeze(-1)) / temp
            )
        else:
            # 推理: hard top-k
            k_int = max(2, int(torch.sigmoid(self.sparsity_logit).item()
                               * num_nodes))
            _, topk_idx = scores.topk(k_int, dim=-1)
            soft_mask = torch.zeros_like(scores)
            soft_mask.scatter_(-1, topk_idx, 1.0)

        # 与predefined mask相交
        combined = soft_mask * base_mask.to(scores.device)
        return combined * scores

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            base_mask = self.mask[index]
            masked_adj = self._gumbel_sinkhorn_topk(a, base_mask)
            result.append(masked_adj)

        _dbg("mask.sparsity_k",
             f"k_ratio={torch.sigmoid(self.sparsity_logit).item():.3f}",
             "graph")
        _dbg("mask.gs_temperature",
             self.gs_temperature, "graph")

        return result
