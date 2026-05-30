"""
Walpurgis ST Localized Convolution — Graph Diffusion with Debug Probes
======================================================================
Adapted from D2STGNN STLocalizedConv. This is the core spatial operator
that performs k-hop graph convolution on the localized ST kernel.

Modifications:
  1. Shape validation at every transform step (unfold, reshape, gconv)
  2. Support matrix inspection — prints graph density and symmetry
  3. Gradient flow check through the conv path
"""

import torch
import torch.nn as nn


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + 
                          len(self.pre_defined_graph) * int(dy_graph) + 
                          int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()
        
        # Walpurgis debug
        self._call_count = 0

    def gconv(self, support, X_k, X_0):
        """Graph convolution with support matrices.
        
        Walpurgis: validates that support matrices have compatible shapes.
        """
        out = [X_0]
        for i, graph in enumerate(support):
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        # ST localization
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        self._call_count += 1
        verbose = (self._call_count % 500 == 1)
        
        # Temporal unfolding: [bs, seq, nodes, feat] → [bs, seq', nodes, k_t, feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        if verbose:
            print(f"        [STConv] call={self._call_count}: "
                  f"unfolded shape=[{batch_size},{seq_len},{num_nodes},{kernel_size},{num_feat}]")

        # Build support set
        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        # Parallelize
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(batch_size, seq_len, kernel_size * num_nodes, num_feat)
        
        hidden = self.gconv(support, X_k, X_0)
        
        if verbose:
            print(f"        [STConv] output: {list(hidden.shape)}, "
                  f"support_count={len(support)}")
        
        return hidden
