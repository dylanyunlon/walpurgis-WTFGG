"""Meridian DynamicGraphConstructor — cosine similarity + adaptive threshold.
Changes vs upstream:
  - Distance: cosine similarity with temperature (upstream: Q*K^T attention)
  - Mask: adaptive threshold per-sample (upstream: fixed predefined mask)
  - Normalizer: symmetric D^{-1/2}AD^{-1/2} (upstream: row-normalize D^{-1}A)
  - Debug: prints graph sparsity and edge statistics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, sys, os

from .utils import Mask, Normalizer, MultiOrder

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class CosineDistanceFunction(nn.Module):
    """Cosine similarity with learnable temperature for graph construction."""
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.input_seq_len = model_args['seq_length']
        self.dropout = nn.Dropout(model_args['dropout'])

        # temporal feature extraction
        self.fc_ts1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # projection to shared space
        feat_dim = self.hidden_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.proj_u = nn.Linear(feat_dim, self.hidden_dim, bias=False)
        self.proj_d = nn.Linear(feat_dim, self.hidden_dim, bias=False)
        # learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts2(self.dropout(self.bn(F.relu(self.fc_ts1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        emb_d = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb_u = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        feat_u = torch.cat([dy_feat, T_D, D_W, emb_u], dim=-1)
        feat_d = torch.cat([dy_feat, T_D, D_W, emb_d], dim=-1)

        # project and cosine similarity
        z_u = F.normalize(self.proj_u(feat_u), dim=-1)
        z_d = F.normalize(self.proj_d(feat_d), dim=-1)
        temp = F.softplus(self.temperature) + 0.1
        sim_ud = torch.bmm(z_u, z_d.transpose(-1, -2)) / temp
        sim_du = torch.bmm(z_d, z_u.transpose(-1, -2)) / temp
        W_ud = torch.softmax(sim_ud, dim=-1)
        W_du = torch.softmax(sim_du, dim=-1)

        if _DBG:
            sparsity_ud = (W_ud < 0.01).float().mean().item()
            print(f"[MER:cos_dist] temp={temp.item():.3f} sparsity={sparsity_ud:.3f} "
                  f"sim_range=[{sim_ud.min().item():.3f},{sim_ud.max().item():.3f}]",
                  file=sys.stderr)

        return [W_ud, W_du]


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']

        self.distance_function = CosineDistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
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
        return dynamic_graphs
