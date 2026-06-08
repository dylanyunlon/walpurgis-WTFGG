"""
Corona STLocalizedConv — 算法改写:
  upstream: 直接图卷积 gconv
  corona: attention-weighted graph conv — 在gconv中加入可学习的
          注意力权重对不同阶图矩阵加权, 替代均等对待
"""
import torch
import torch.nn as nn
from ... import _dbg, dataflow_checkpoint


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
                          len(self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self._gcn_updt = None  # lazy init based on actual support size
        self._expected_in = self.hidden_dim * self.num_matric

        # Corona改写: 注意力权重为每个support graph学习一个标量权重
        self.graph_attn_weights = nn.Parameter(torch.ones(self.num_matric - 1))

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        # Corona改写: 用softmax归一化的注意力权重加权各阶图卷积
        n_support = len(support)
        if n_support > 0:
            raw_weights = self.graph_attn_weights[:min(n_support, len(self.graph_attn_weights))]
            if n_support > len(raw_weights):
                # pad with zeros if more support graphs than expected
                pad = torch.zeros(n_support - len(raw_weights), device=raw_weights.device)
                raw_weights = torch.cat([raw_weights, pad])
            attn = torch.softmax(raw_weights, dim=0)
        for idx, graph in enumerate(support):
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            if n_support > 0:
                out.append(H_k * attn[idx])
            else:
                out.append(H_k)
        out = torch.cat(out, dim=-1)
        # Lazy init gcn_updt to match actual concat size
        actual_dim = out.shape[-1]
        if self._gcn_updt is None or self._gcn_updt.in_features != actual_dim:
            self._gcn_updt = nn.Linear(actual_dim, self.hidden_dim).to(out.device)
        out = self._gcn_updt(out)
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
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(batch_size, seq_len, kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        dataflow_checkpoint("corona_st_conv.out", hidden)
        return hidden
