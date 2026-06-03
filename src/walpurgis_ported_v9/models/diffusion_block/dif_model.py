"""
dif_model.py — v9 port (STLocalizedConv)
Algo delta:
  1. BatchNorm2d → GroupNorm(4 groups), 对小 batch 更稳定
  2. gconv 输出加残差 skip: out = gcn(X_k, X_0) + X_0
  3. fc_list_updt 后加第二层非线性 Mish (x·tanh(softplus(x)))
  4. get_graph: 高阶 power 后用 fill_diagonal_(0) 替代 mask 矩阵乘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v9 import _dbg

_TAG = "dif_model"


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
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph)
                          + len(self.pre_defined_graph) * int(dy_graph)
                          + int(sta_graph)) * self.k_s + 1

        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        # v9: second projection with Mish
        self.fc_mish = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        # v9: GroupNorm instead of BatchNorm
        n_groups = min(4, hidden_dim)
        self.gn = nn.GroupNorm(n_groups, hidden_dim)
        self.activation = nn.ReLU()

    def _mish(self, x):
        return x * torch.tanh(F.softplus(x))

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
        # v9: residual skip
        out = out + X_0
        _dbg(_TAG, f"gconv  |out|={out.abs().mean().item():.6g}  support_len={len(support)}")
        return out

    def get_graph(self, support):
        graph_ordered = []
        for graph in support:
            k_1_order = graph.clone()
            # v9: fill_diagonal_ instead of mask multiply
            k_1_order.fill_diagonal_(0)
            graph_ordered.append(k_1_order)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                k_high = k_1_order.clone()
                k_high.fill_diagonal_(0)
                graph_ordered.append(k_high)

        st_local_graph = []
        for graph in graph_ordered:
            g = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            g = g.reshape(g.shape[0], g.shape[1] * g.shape[2])
            st_local_graph.append(g)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, S, N, K, D = X.shape

        support = []
        if self.use_predefined_graph:
            support += self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support += dynamic_graph
        if self.use_static__hidden_graph:
            support += self.get_graph(static_graph)

        X_flat = X.reshape(B, S, N, K * D)
        out = self.fc_list_updt(X_flat)
        out = self.activation(out)
        # v9: Mish second layer
        out = self._mish(self.fc_mish(out))

        out = out.view(B, S, N, K, D)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, S, K * N, D)
        hidden = self.gconv(support, X_k, X_0)

        # v9: GroupNorm (need [B, C, ...] layout)
        h_perm = hidden.permute(0, 3, 1, 2)  # [B, D, S, N]
        h_perm = self.gn(h_perm)
        hidden = h_perm.permute(0, 2, 3, 1)  # back to [B, S, N, D]

        _dbg(_TAG, f"STConv out  shape={list(hidden.shape)}  "
                    f"mean={hidden.mean().item():.6g}  std={hidden.std().item():.6g}")
        return hidden
