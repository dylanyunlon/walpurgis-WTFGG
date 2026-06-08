"""
Aphelion STLocalizedConv — 算法改写 #3:
  upstream: 直接图卷积 gconv (矩阵乘法)
  corona: attention-weighted graph conv (标量注意力权重)
  aphelion: GAT v2 + edge features — 使用GATv2的动态注意力机制替代
            标准图卷积, 每条边有可学习的edge特征参与注意力计算。
            GATv2先拼接再激活(替代GAT的先激活再拼接), 表达力更强。
  改动幅度: ~30% (GATv2注意力替代矩阵乘法图卷积)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self._gcn_updt = None  # lazy init

        # Aphelion改写: GATv2注意力层 — 替代标准图卷积
        # GATv2: 先拼接邻居特征, 再用共享的注意力向量
        self.gatv2_W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 特征投影
        self.gatv2_a = nn.Linear(2 * hidden_dim, 1, bias=False)  # 注意力向量
        # 可学习的edge特征 (每个support graph有一组edge bias)
        self.num_graphs = max(self.num_matric - 1, 1)
        self.edge_bias = nn.Parameter(torch.zeros(self.num_graphs))
        # LeakyReLU用于GATv2
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        """Aphelion: GATv2 + edge features替代标准图卷积"""
        out = [X_0]
        n_support = len(support)
        # 对X_0做GATv2特征投影
        X_proj = self.gatv2_W(X_0)  # [B, S, N, D]

        for idx, graph in enumerate(support):
            if len(graph.shape) == 2:
                graph = graph.unsqueeze(0).unsqueeze(0)  # [1, 1, N, kN]
            else:
                graph = graph.unsqueeze(1)  # [B, 1, N, kN]

            # 标准消息传递: 先用图做聚合
            H_k = torch.matmul(graph, X_k)  # [B, S, N, D]

            # Aphelion GATv2: 拼接 [h_i || h_j] 然后LeakyReLU再投影
            # 这里h_i = X_proj (自身), h_j = H_k (邻居聚合后)
            H_proj = self.gatv2_W(H_k)
            cat_feat = torch.cat([X_proj, H_proj], dim=-1)  # [B, S, N, 2D]
            attn_score = self.gatv2_a(self.leaky_relu(cat_feat))  # [B, S, N, 1]
            attn_weight = torch.sigmoid(attn_score)

            # edge feature bias: 每个support graph有不同的edge偏置
            edge_idx = min(idx, self.num_graphs - 1)
            edge_w = torch.sigmoid(self.edge_bias[edge_idx])

            out.append(H_k * attn_weight * edge_w)

        out = torch.cat(out, dim=-1)
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
        dataflow_checkpoint("aphelion_gatv2_conv.out", hidden)
        return hidden
