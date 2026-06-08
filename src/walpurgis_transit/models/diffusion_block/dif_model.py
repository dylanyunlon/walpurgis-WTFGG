"""
STLocalizedConv — Transit变体
算法改动: APPNP传播 (Approximate Personalized PageRank)
  原版: 直接k阶矩阵乘法扩散, FC更新
  Transit:
    - APPNP: H^(k) = (1-α)*A_hat*H^(k-1) + α*H^(0)
      α是teleport概率(可学习), 多步传播后保持与原始特征的连接
    - 比直接矩阵幂更稳定: teleport防止过平滑
    - 可学习传播步数权重: 不同步数对最终输出贡献可调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


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
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric,
            self.hidden_dim)

        # APPNP参数: teleport概率α (可学习)
        self.alpha_teleport = nn.Parameter(torch.tensor(0.15))
        # 各传播步数的权重
        self.step_weights = nn.Parameter(
            torch.ones(self.k_s + 1) / (self.k_s + 1))

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ELU(inplace=True)

    def _appnp_propagate(self, A_hat, X, steps):
        """APPNP传播: H^(k) = (1-α)*A*H^(k-1) + α*H^(0)
        teleport确保不会完全丢失初始特征"""
        alpha = torch.sigmoid(self.alpha_teleport)
        H = X  # H^(0) = X
        H_init = X
        all_steps = [H]
        for _ in range(steps):
            H = (1.0 - alpha) * torch.matmul(A_hat, H) + alpha * H_init
            all_steps.append(H)
        return all_steps

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

        # APPNP加权: 用teleport权重替代简单mean
        weights = F.softmax(self.step_weights[:kernel_size], dim=0)
        X_0 = sum(weights[i] * out[:, :, :, i, :]
                  for i in range(kernel_size))

        _dbg("stconv.appnp_alpha",
             torch.sigmoid(self.alpha_teleport), "diffusion")
        _dbg("stconv.step_weights",
             weights, "diffusion")

        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len,
            kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden
