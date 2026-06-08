"""Prism dynamic graph constructor: adds contrastive-aware edge re-weighting.
After standard distance + mask + normalize, Prism applies a contrastive
modulation that upweights edges between nodes with similar embeddings
and downweights edges between dissimilar nodes."""
import torch
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder


class ContrastiveEdgeModulator(nn.Module):
    """Prism特有: 基于节点embedding相似度调制边权重"""
    def __init__(self, node_dim):
        super().__init__()
        self.modulation_strength = nn.Parameter(
            torch.tensor(0.1))

    def forward(self, adj_list, E_d, E_u):
        # E_d, E_u: [N, d]
        # 计算节点间余弦相似度
        E_combined = (E_d + E_u) / 2
        E_norm = torch.nn.functional.normalize(
            E_combined, dim=-1)
        sim = torch.mm(E_norm, E_norm.t())  # [N, N]
        # 调制强度
        strength = torch.sigmoid(self.modulation_strength)
        # 对每个邻接矩阵施加相似度调制
        modulated = []
        for adj in adj_list:
            if adj.dim() == 2:
                # static: [N, N]
                mod = adj * (1 + strength * sim)
                modulated.append(mod)
            else:
                # dynamic: [B, N, N]
                sim_expanded = sim.unsqueeze(0).expand_as(adj)
                mod = adj * (1 + strength * sim_expanded)
                modulated.append(mod)
        return modulated


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
        # Prism特有: 对比感知边调制
        self.edge_modulator = ContrastiveEdgeModulator(
            self.node_dim)

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
        # Prism特有: 对比感知边调制
        dist_mx = self.edge_modulator(
            dist_mx, E_d, E_u)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs
