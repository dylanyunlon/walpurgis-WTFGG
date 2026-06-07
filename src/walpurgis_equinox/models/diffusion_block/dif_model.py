import torch
import torch.nn as nn
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:stconv:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class STLocalizedConv(nn.Module):
    """upstream: BN+ReLU, gconv无skip
    equinox: WeightNorm + Mish激活, gconv加残差skip, 对角线清零"""
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None,
                 dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph
        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        n_dy_graphs = 2
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) +
                          n_dy_graphs * int(dy_graph) * self.k_s +
                          int(sta_graph) * self.k_s) + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        # upstream: BatchNorm2d + ReLU
        # equinox: WeightNorm + Mish (batch无关, 小batch更稳定)
        self.wn_proj = nn.utils.weight_norm(nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim))
        self.activation = nn.Mish()

        # equinox: gconv残差skip
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

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
        # equinox: 残差skip
        out = out + self.skip_proj(X_0)
        _edbg("gconv_out", out)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                k_1_order = k_1_order * mask
                graph_ordered.append(k_1_order * mask)
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, seq_len, N, K, D = X.shape
        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(B, seq_len, N, K * D)
        out = self.fc_list_updt(X)
        # equinox: WeightNorm + Mish
        out = self.wn_proj(out)
        out = self.activation(out)
        out = out.view(B, seq_len, N, K, D)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, seq_len, K * N, D)
        hidden = self.gconv(support, X_k, X_0)
        _edbg("stconv_out", hidden)
        return hidden
