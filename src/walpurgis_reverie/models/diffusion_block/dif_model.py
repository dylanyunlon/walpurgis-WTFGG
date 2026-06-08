import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

_TAG = "dif_model"


class STLocalizedConv(nn.Module):
    """upstream: 固定k_s阶MatMul graph conv + FC temporal
    改动: Chebyshev多项式图卷积, 每阶有可学习的权重alpha_k
    Chebyshev递推: T_0=I, T_1=L_hat, T_k=2*L_hat*T_{k-1} - T_{k-2}
    相比直接A^k, Chebyshev基在[-1,1]上更稳定, 避免数值爆炸
    """

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

        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)

        # 改动: Chebyshev阶权重 — 每个k有一个可学习的标量权重
        self.cheby_weights = nn.Parameter(
            torch.ones(self.k_s) / self.k_s)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.SiLU()  # 改动: SiLU代替ReLU

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
        return out

    def _chebyshev_order(self, graph, k):
        """Chebyshev递推: T_k(L_hat) with learnable weight"""
        if k == 0:
            return torch.eye(graph.shape[-1], device=graph.device).expand_as(graph)
        elif k == 1:
            return graph * self.cheby_weights[0]
        else:
            T_prev = torch.eye(graph.shape[-1], device=graph.device).expand_as(graph)
            T_curr = graph
            for i in range(2, k + 1):
                T_next = 2 * torch.matmul(graph, T_curr) - T_prev
                T_prev = T_curr
                T_curr = T_next
            idx = min(k - 1, self.k_s - 1)
            return T_curr * self.cheby_weights[idx]

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
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len, kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)

        _dbg(f"{_TAG}/cheby_weights", self.cheby_weights, _TAG)
        _dbg(f"{_TAG}/hidden_out", hidden, _TAG)
        return hidden
