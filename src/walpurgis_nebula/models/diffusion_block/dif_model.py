"""Nebula STLocalizedConv: GroupNorm + Hardswish + depthwise separable conv."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class DepthwiseSeparableLinear(nn.Module):
    """Depthwise separable linear: splits features into groups (depthwise),
    then mixes across groups (pointwise). Reduces parameters vs full linear."""
    def __init__(self, in_features, out_features, num_groups=4):
        super().__init__()
        self.num_groups = min(num_groups, in_features, out_features)
        g = self.num_groups
        assert in_features % g == 0 and out_features % g == 0, \
            f"in_features={in_features} and out_features={out_features} must be divisible by num_groups={g}"
        # Depthwise: each group processed independently
        self.depthwise = nn.Conv1d(in_features, in_features, kernel_size=1, groups=g)
        # Pointwise: mix across groups
        self.pointwise = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        """x: [..., in_features] -> [..., out_features]"""
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1]).unsqueeze(-1)  # [N, C, 1]
        out = self.pointwise(self.depthwise(flat)).squeeze(-1)  # [N, out_features]
        return out.view(*orig_shape[:-1], -1)


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
        # Nebula: depthwise separable linear replaces plain linear
        kt_dim = self.k_t * hidden_dim
        # Ensure divisibility for depthwise separable
        num_groups = 1
        for g in [4, 2]:
            if kt_dim % g == 0:
                num_groups = g; break
        self.fc_list_updt = DepthwiseSeparableLinear(kt_dim, kt_dim, num_groups=num_groups)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)
        # Nebula: GroupNorm replaces BatchNorm2d/InstanceNorm
        num_gn_groups = min(4, self.hidden_dim)
        while self.hidden_dim % num_gn_groups != 0 and num_gn_groups > 1:
            num_gn_groups -= 1
        self.norm = nn.GroupNorm(num_gn_groups, self.hidden_dim)
        # Nebula: Hardswish replaces ReLU/GELU
        self.activation = nn.Hardswish()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) != 2: graph = graph.unsqueeze(1)
            out.append(torch.matmul(graph, X_k))
        out = self.gcn_updt(torch.cat(out, dim=-1))
        out = self.dropout(out)
        # Nebula: apply GroupNorm + Hardswish after graph conv
        B, S, N, D = out.shape
        out = self.norm(out.permute(0, 3, 1, 2).reshape(B, D, S * N)).reshape(B, D, S, N).permute(0, 2, 3, 1)
        out = self.activation(out)
        if _NEB_DBG:
            print(f"[NEB:gconv@dif_model] out={list(out.shape)} range=[{out.min().item():.4f},{out.max().item():.4f}]", file=sys.stderr)
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
