"""Flux DynamicGraphConstructor: 流式感知的动态图构建.
与upstream(用全序列最后步构建图)和vortex(同upstream)不同,
Flux使用滑动窗口内的均值特征构建动态图,
使得图结构更稳定, 不会因单步噪声剧烈变化."""
import torch.nn as nn
import sys
import os

from .utils import DistanceFunction, Mask, Normalizer, \
    MultiOrder

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.distance_function = DistanceFunction(
            **model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)
        # Flux: 流式窗口大小 for graph construction
        self._graph_window = model_args.get(
            'seq_length', 12)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0],
                    k_order_graph.shape[1],
                    k_order_graph.shape[2] *
                    k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']
        # distance calculation
        dist_mx = self.distance_function(
            X, E_d, E_u, T_D, D_W)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)
        if _FX_DBG:
            print(f"[FX:dy_graph] n_graphs="
                  f"{len(dynamic_graphs)} "
                  f"input_seq={X.shape[1]}",
                  file=sys.stderr)
        return dynamic_graphs
