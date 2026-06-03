import torch
import torch.nn as nn
import sys

_DBG = ("--dbg" in sys.argv)


class Mask(nn.Module):
    """算法改动: top-k sparse masking
    原版: 用 predefined adj 做 elementwise 乘法 (硬 mask)
    改为: 先做 predefined adj mask, 然后对每一行只保留 top-k 个值,
          其余置零. k = min(num_nodes, 2 * avg_degree_of_predefined_graph)
    这样动态图既参考了先验拓扑, 又保持了稀疏性, 防止全连接退化
    """

    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 从 predefined graph 估计平均度
        with torch.no_grad():
            avg_deg = (self.mask[0] > 0).float().sum(dim=-1).mean().item()
        self.top_k = max(int(2 * avg_deg), 5)
        if _DBG:
            print(f"[DBG][Mask] avg_degree={avg_deg:.1f}  top_k={self.top_k}",
                  flush=True)

    def _mask(self, index, adj):
        prior = self.mask[index] + torch.ones_like(
            self.mask[index]) * 1e-7
        masked_adj = prior.to(adj.device) * adj

        # top-k sparsification per row
        num_nodes = masked_adj.shape[-1]
        k = min(self.top_k, num_nodes)
        # masked_adj: [B, N, N]
        topk_vals, topk_idx = torch.topk(masked_adj, k, dim=-1)
        sparse_adj = torch.zeros_like(masked_adj)
        sparse_adj.scatter_(-1, topk_idx, topk_vals)
        return sparse_adj

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._mask(index, a))
        if _DBG:
            with torch.no_grad():
                nnz = (result[0] > 1e-6).float().mean().item()
                print(f"[DBG][Mask] post-mask nonzero_frac={nnz:.4f}  "
                      f"top_k={self.top_k}", flush=True)
        return result
