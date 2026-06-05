"""
STLocalizedConv — Nightfall变体
算法改写:
  1. gconv输出加残差skip: out = gconv(X) + X_0 (梯度短路)
  2. ReLU → SiLU (Swish, 更平滑且自门控)
  3. BatchNorm → GroupNorm (对batch size不敏感)
"""
import torch
import torch.nn as nn
from ... import _dbg


class STLocalizedConv(nn.Module):
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
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) +
                          len(self.pre_defined_graph) * int(dy_graph) +
                          int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)
        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)
        # GroupNorm替代BatchNorm (num_groups=4)
        num_groups = min(4, hidden_dim)
        self.gn = nn.GroupNorm(num_groups, self.hidden_dim)
        # SiLU替代ReLU
        self.activation = nn.SiLU()

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
        # 残差skip: 加回X_0
        out = out + X_0
        _dbg("st_conv.gconv_out", out, "model")
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
        out = self.activation(out)  # SiLU
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len, kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        # GroupNorm (需要reshape到 [B*T, C, N] 格式)
        B, T, N, C = hidden.shape
        hidden = hidden.reshape(B * T, C, N).contiguous()
        hidden = self.gn(hidden)
        hidden = hidden.reshape(B, T, N, C)
        return hidden
