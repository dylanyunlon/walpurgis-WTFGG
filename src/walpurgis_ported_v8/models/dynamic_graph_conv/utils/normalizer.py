import torch
import torch.nn as nn
import sys

from utils.cal_adj import remove_nan_inf

_DBG = ("--dbg" in sys.argv)


class Normalizer(nn.Module):
    """算法改动: symmetric normalization D^{-1/2} A D^{-1/2}
    原版: row normalization D^{-1} A
    改为: symmetric normalization, 保持特征值在 [-1, 1],
          训练时梯度更稳定 (spectral 意义上)
    """

    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        degree = remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        # D^{-1/2}
        D_inv_sqrt = torch.diag_embed(degree)
        # D^{-1/2} A D^{-1/2}
        normed = torch.bmm(torch.bmm(D_inv_sqrt, graph), D_inv_sqrt)
        return normed

    def forward(self, adj):
        result = [self._norm(a) for a in adj]
        if _DBG:
            with torch.no_grad():
                for i, r in enumerate(result):
                    print(f"[DBG][Normalizer] adj[{i}] "
                          f"mean={r.mean().item():.5f}  "
                          f"max={r.max().item():.5f}", flush=True)
        return result


class MultiOrder(nn.Module):
    """算法改动: 高阶衰减因子
    原版: A^k 直接乘 mask 就完了
    改为: A^k 乘以 decay^(k-1), decay=0.5
          高阶邻居的贡献随阶数指数衰减, 避免过度平滑 (over-smoothing)
    """

    def __init__(self, order=2, decay=0.5):
        super().__init__()
        self.order = order
        self.decay = decay

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            # 算法改动: decay factor for higher orders
            scale = self.decay ** (k - 1)
            graph_ordered.append(k_1_order * mask * scale)
        return graph_ordered

    def forward(self, adj):
        result = [self._multi_order(a) for a in adj]
        if _DBG:
            with torch.no_grad():
                for i, orders in enumerate(result):
                    for k, g in enumerate(orders):
                        print(f"[DBG][MultiOrder] adj[{i}] order={k+1}  "
                              f"mean={g.mean().item():.6f}  "
                              f"decay_scale={self.decay**k:.3f}", flush=True)
        return result
