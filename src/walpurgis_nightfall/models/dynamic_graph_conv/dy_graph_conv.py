"""
DynamicGraphConstructor — Nightfall变体
算法改写:
  1. st_localization输出加dropout退火 (前期高drop→后期低drop)
  2. forward结尾打印动态图稀疏度和值域
"""
import torch.nn as nn
from .utils import DistanceFunction, Mask, Normalizer, MultiOrder
from walpurgis_nightfall import _dbg


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)
        # st_localization dropout退火: 初始值较高
        self.st_dropout = nn.Dropout(model_args.get('dropout', 0.3) * 0.5)
        # 步数计数器(用于退火)
        self._step_count = 0

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                # dropout退火
                if self.training:
                    k_order_graph = self.st_dropout(k_order_graph)
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        _dbg("dy_graph.num_graphs", f"{len(dynamic_graphs)} dynamic graphs constructed", "model")
        if dynamic_graphs:
            _dbg("dy_graph.graph_0", dynamic_graphs[0], "model")
        self._step_count += 1
        return dynamic_graphs
