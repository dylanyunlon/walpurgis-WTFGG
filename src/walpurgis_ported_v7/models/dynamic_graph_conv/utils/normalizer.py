import torch
import torch.nn as nn
import sys

from utils.cal_adj import remove_nan_inf

_DBG_NORM = ("--dbg-norm" in sys.argv)


class Normalizer(nn.Module):
    """算法改动: 从 row-normalization (D^{-1}A) 改为
    symmetric normalization (D^{-1/2} A D^{-1/2}),
    这样得到的矩阵是对称的, 谱性质更好, GCN 论文本来就推荐这个。
    """
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        # 算法改动: D^{-1/2} 而非 D^{-1}
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        d_mat = torch.diag_embed(d_inv_sqrt)
        # D^{-1/2} A D^{-1/2}
        normed_graph = torch.bmm(torch.bmm(d_mat, graph), d_mat)

        if _DBG_NORM:
            with torch.no_grad():
                sym_err = (normed_graph - normed_graph.transpose(-1, -2)).abs().max().item()
                print(f"[DBG-NORM] symmetric_err={sym_err:.6f}  "
                      f"spectral_radius~{normed_graph.abs().max().item():.4f}")

        return normed_graph

    def forward(self, adj):
        return [self._norm(a) for a in adj]


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
            # 算法改动: 对高阶矩阵做 row-sum renormalization
            # 防止高阶 power 导致数值膨胀
            row_sum = k_1_order.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            k_1_order_normed = k_1_order / row_sum
            graph_ordered.append(k_1_order_normed * mask)

            if _DBG_NORM:
                with torch.no_grad():
                    print(f"[DBG-NORM] order-{k}  "
                          f"pre_renorm_max={k_1_order.abs().max().item():.4f}  "
                          f"post_renorm_max={k_1_order_normed.abs().max().item():.4f}")
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
