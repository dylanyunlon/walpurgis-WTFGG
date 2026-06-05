import torch
import torch.nn as nn
from walpurgis_walking import _dbg

_TAG = "dif_conv"


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None, sta_graph=None, **model_args):
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

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        # 改动1: InstanceNorm 替代 BN — 对每个样本独立归一化
        self.norm = nn.InstanceNorm2d(self.hidden_dim, affine=True)
        # 改动2: GELU 替代 ReLU
        self.activation = nn.GELU()

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
        out = self.gcn_updt(out)
        out = self.dropout(out)
        # 改动3: gconv 内加 skip connection — upstream 无
        out = out + X_0
        _dbg(_TAG, "gconv_skip", out_norm=out.norm(), x0_norm=X_0.norm())
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
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        _dbg(_TAG, "input", X=X)
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, L, N, K, D = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(B, L, N, K * D)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(B, L, N, K, D)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, L, K * N, D)
        hidden = self.gconv(support, X_k, X_0)

        # 改动4: InstanceNorm — reshape (B,L,N,D)→(B*L,D,N,1) for IN2d
        hs = hidden.shape
        h_flat = hidden.reshape(hs[0] * hs[1], hs[2], hs[3]).permute(0, 2, 1).unsqueeze(-1)
        h_flat = self.norm(h_flat)
        hidden = h_flat.squeeze(-1).permute(0, 2, 1).reshape(hs)

        _dbg(_TAG, "output", hidden=hidden)
        return hidden
