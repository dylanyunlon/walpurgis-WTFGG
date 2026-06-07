"""Eclipse STLocalizedConv: InstanceNorm2d + GELU + gconv residual skip."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']; self.k_t = model_args['k_t']; self.hidden_dim = hidden_dim
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre; self.use_dynamic_hidden_graph = dy_graph; self.use_static__hidden_graph = sta_graph
        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)
        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)
        self.norm = nn.InstanceNorm2d(self.hidden_dim, affine=True)
        self.activation = nn.GELU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) != 2: graph = graph.unsqueeze(1)
            out.append(torch.matmul(graph, X_k))
        out = self.gcn_updt(torch.cat(out, dim=-1))
        out = self.dropout(out) + X_0  # residual skip
        if _ECL_DBG: print(f"[ECL:gconv@dif_model] out={list(out.shape)} range=[{out.min().item():.4f},{out.max().item():.4f}]", file=sys.stderr)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph; graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order); graph_ordered.append(k_1_order * mask)
        st_local = []
        for g in graph_ordered:
            g = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            st_local.append(g.reshape(g.shape[0], g.shape[1] * g.shape[2]))
        return st_local

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, S, N, K, F = X.shape
        support = []
        if self.use_predefined_graph: support += self.pre_defined_graph
        if self.use_dynamic_hidden_graph: support += dynamic_graph
        if self.use_static__hidden_graph: support += self.get_graph(static_graph)
        X = X.reshape(B, S, N, K * F)
        out = self.activation(self.fc_list_updt(X))
        out = out.view(B, S, N, K, F)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, S, K * N, F)
        return self.gconv(support, X_k, X_0)
