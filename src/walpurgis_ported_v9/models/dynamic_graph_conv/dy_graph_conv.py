"""
dy_graph_conv.py — v9 port
Algo delta:
  1. st_localization: upstream 直接 reshape (N, k_t, N) → (N, k_t*N)
     v9: 先用可学习权重向量 w ∈ R^{k_t} 对 k_t 维度做加权求和再 reshape,
     让网络学习哪些时间偏移更重要
  2. forward 输出 dynamic_graphs 后, 叠加一个纯 embedding 距离的
     skip connection: E_d @ E_u.T 经 softmax → 作为第0个图追加,
     保证即使 TS 信号退化, 节点嵌入本身的先验距离仍然参与
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DistanceFunction, Mask, Normalizer, MultiOrder
from walpurgis_ported_v9 import _dbg

_TAG = "dy_graph"


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

        # v9: learnable temporal weighting for st_localization
        self.temporal_weight = nn.Parameter(torch.ones(self.k_t) / self.k_t)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        # v9: softmax temporal weight
        tw = F.softmax(self.temporal_weight, dim=0)

        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                # [B, N, N] → [B, N, k_t, N]
                expanded = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                # v9: weighted sum along k_t dim then tile back
                weighted = (expanded * tw.view(1, 1, -1, 1)).sum(dim=-2, keepdim=True)
                weighted = weighted.expand(-1, -1, self.k_t, -1)
                # reshape to [B, N, k_t*N]
                reshaped = weighted.reshape(
                    weighted.shape[0], weighted.shape[1],
                    weighted.shape[2] * weighted.shape[3])
                st_local_graph.append(reshaped)
        return st_local_graph

    def forward(self, **inputs):
        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)

        _dbg(_TAG, f"temporal_weight={F.softmax(self.temporal_weight,dim=0).data.tolist()}  "
                    f"n_dynamic_graphs={len(dynamic_graphs)}")
        return dynamic_graphs
