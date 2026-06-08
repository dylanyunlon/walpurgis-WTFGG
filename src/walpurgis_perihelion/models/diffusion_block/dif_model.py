"""
STLocalizedConv — Perihelion变体
算法改动: GraphSAGE + Jumping Knowledge
  原版: 直接k阶矩阵乘法扩散, FC更新
  Perihelion:
    - GraphSAGE风格聚合: 对每阶邻居做mean-aggregate再concat自身
      区别于全阶矩阵乘幂的扩散, SAGE先聚合再变换
    - Jumping Knowledge: 收集每阶的中间表示, 用LSTM attention
      选择性地组合不同阶的信息(而非仅用最终阶)
    - 每阶独立的可学习权重矩阵(不共享)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


class GraphSAGEAggregator(nn.Module):
    """GraphSAGE均值聚合器: aggregate neighbors → concat self → transform"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # SAGE: concat(self, mean_neighbor) → linear
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X, adj):
        """
        X: [*, N, D] 节点特征 (caller负责reshape到3D)
        adj: [N, N] 邻接矩阵 (caller负责提取子块)
        """
        # 行归一化获取均值
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / deg
        # matmul: adj [N,N] x X [..., N, D] → [..., N, D]
        neigh_feat = torch.matmul(adj_norm, X)

        # SAGE: transform(self) + transform(neighbor)
        out = self.self_linear(X) + self.neigh_linear(neigh_feat)
        out = self.norm(out)
        out = F.relu(out)
        return out


class JumpingKnowledge(nn.Module):
    """Jumping Knowledge: 用注意力机制选择性组合各阶表示"""

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # 注意力打分: 每阶表示 → 标量权重
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.num_layers = num_layers

    def forward(self, layer_outputs):
        """
        layer_outputs: list of [B, L, N, D], 每阶一个
        返回: [B, L, N, D] 加权组合
        """
        stacked = torch.stack(layer_outputs, dim=0)  # [K, B, L, N, D]
        # 计算每阶的注意力权重
        scores = self.attn_score(stacked)  # [K, B, L, N, 1]
        weights = F.softmax(scores, dim=0)

        _dbg("jk.layer_weights",
             weights.squeeze(-1).mean(dim=(1, 2, 3)),
             "diffusion")

        combined = (stacked * weights).sum(dim=0)
        return combined


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None,
                 sta_graph=None, **model_args):
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
        self.num_matric = (
            int(use_pre) * len(self.pre_defined_graph)
            + len(self.pre_defined_graph) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(
            self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim,
            self.k_t * hidden_dim, bias=False)

        # GraphSAGE聚合器: 每阶图卷积独立
        self.sage_layers = nn.ModuleList([
            GraphSAGEAggregator(hidden_dim, hidden_dim)
            for _ in range(self.k_s)
        ])

        # Jumping Knowledge: 组合各阶表示
        self.jk = JumpingKnowledge(hidden_dim, self.k_s)

        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric,
            self.hidden_dim)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

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

        dataflow_checkpoint("stconv.gconv_out", out)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(
            support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(
                -1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0],
                graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        dataflow_checkpoint("stconv.input", X)
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(batch_size, seq_len, num_nodes,
                       kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes,
                       kernel_size, num_feat)

        # GraphSAGE + Jumping Knowledge:
        # 收集每阶SAGE的输出, 用JK组合
        sage_outputs = []
        # 基础表示: 对kernel维均值
        base_repr = out.mean(dim=3)  # [B, L, N, D]
        sage_outputs.append(base_repr)

        current = base_repr
        for sage_layer in self.sage_layers:
            # 使用第一个support图做邻居聚合
            if len(support) > 0:
                adj = support[0]
                B, L, N, D = current.shape
                flat = current.reshape(B * L, N, D)
                # 提取N×N子块, 无论adj是2D还是3D
                if adj.dim() == 2:
                    adj_small = adj[:N, :N]
                else:
                    # 3D: [B, N*k_t, N*k_t] → 取第一个batch的N×N
                    adj_small = adj[0, :N, :N]
                agg = sage_layer(flat, adj_small)
                current = agg.reshape(B, L, N, D)
            sage_outputs.append(current)

        # Jumping Knowledge注意力聚合
        X_0 = self.jk(sage_outputs)

        _dbg("stconv.sage_layers_used",
             f"{len(self.sage_layers)}", "diffusion")

        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len,
            kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden
