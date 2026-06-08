import torch
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder


class DynamicGraphConstructor(nn.Module):
    """Helix改写: 在dynamic graph构建后加入top-k稀疏化,
    与model.py中的static graph稀疏化配合"""
    def __init__(self, **model_args):
        super().__init__()
        # model args
        self.k_s = model_args['k_s']  # spatial order
        self.k_t = model_args['k_t']  # temporal kernel size
        # hidden dimension of
        self.hidden_dim = model_args['num_hidden']
        # trainable node embedding dimension
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        # Helix特有: 动态图稀疏化的k比例
        self._dy_topk_ratio = nn.Parameter(torch.tensor(0.7))

    def _topk_sparsify_batch(self, graph):
        """Helix特有: 对batch维度的图做top-k稀疏化"""
        ratio = torch.sigmoid(self._dy_topk_ratio)
        N = graph.shape[-1]
        k = max(1, int(ratio.item() * N))
        topk_vals, topk_idx = torch.topk(graph, k, dim=-1)
        sparse_graph = torch.zeros_like(graph)
        sparse_graph.scatter_(-1, topk_idx, topk_vals)
        # 重新归一化
        row_sum = sparse_graph.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sparse_graph = sparse_graph / row_sum
        return sparse_graph

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        """Dynamic graph learning module.

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            list: dynamic graphs
        """

        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']
        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # Helix特有: 对归一化后的距离矩阵做top-k稀疏化
        dist_mx = [self._topk_sparsify_batch(g) for g in dist_mx]
        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)

        return dynamic_graphs
