"""Spatially-localised temporal convolution — with skip path and GroupNorm.

Changes
-------
1. ``gconv`` — adds a residual skip connection: output = GCN(X) + X_0.
   The original only concatenated then projected; the skip lets gradients
   flow directly through the spatial dimension, which helps on sparser
   graphs (PEMS04/08) where multi-hop diffusion paths are thin.
2. BatchNorm2d → GroupNorm (groups=4).  GroupNorm is batch-size invariant,
   which matters when validation batch size differs from training.
3. ``get_graph`` — mask diagonal uses ``fill_diagonal_`` instead of
   allocating a full eye matrix, saving O(N^2) memory per call.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None, sta_graph=None,
                 **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = (len(self.pre_defined_graph)
                            + int(dy_graph) + int(sta_graph))
        self.num_matric = (
            int(use_pre) * len(self.pre_defined_graph)
            + len(self.pre_defined_graph) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1

        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self._preprocess_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            hidden_dim * self.num_matric, hidden_dim)

        # GroupNorm instead of BatchNorm — batch-size agnostic
        n_groups = min(4, hidden_dim)
        self.norm = nn.GroupNorm(n_groups, hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        projected = self.gcn_updt(out)
        projected = self.dropout(projected)
        # ── residual skip: add X_0 back ──
        result = projected + X_0
        _dbg("STConv.gconv", result)
        return result

    def _preprocess_graph(self, support):
        graph_ordered = []
        for graph in support:
            # in-place diagonal mask — saves an N×N allocation
            mask = torch.ones_like(graph)
            mask.fill_diagonal_(0)
            k_1_order = graph * mask
            graph_ordered.append(k_1_order)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)

        st_local = []
        for g in graph_ordered:
            g = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            g = g.reshape(g.shape[0], g.shape[1] * g.shape[2])
            st_local.append(g)
        return st_local

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, T, N, K, D = X.shape

        support = []
        if self.use_predefined_graph:
            support += self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support += dynamic_graph
        if self.use_static__hidden_graph:
            support += self._preprocess_graph(static_graph)

        X = X.reshape(B, T, N, K * D)
        out = self.activation(self.fc_list_updt(X))
        out = out.view(B, T, N, K, D)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, T, K * N, D)

        hidden = self.gconv(support, X_k, X_0)

        # apply GroupNorm (needs channels-first: B*T, D, N → rearrange)
        BT = B * T
        h_flat = hidden.reshape(BT, N, D).permute(0, 2, 1)   # (BT, D, N)
        h_flat = self.norm(h_flat).permute(0, 2, 1)           # (BT, N, D)
        hidden = h_flat.reshape(B, T, N, D)

        return hidden
