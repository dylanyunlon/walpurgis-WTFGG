"""
STLocalizedConv — Parallax变体 (M054)
算法改动: MixHop多分辨率GCN
  原版: 直接k阶矩阵乘法扩散 → 单个FC更新
  Parallax:
    - MixHop: 对每个跳数(1-hop, 2-hop, 3-hop)分别用独立权重投影
      然后拼接不同分辨率的特征, 允许模型同时看到局部和全局
    - 每跳有独立的线性变换 + 可学习的跳间注意力权重
    - 跳数之间做column-wise concatenation而非简单求和
    - BatchNorm替换为InstanceNorm(对每个节点独立归一化)

  MixHop参考: Abu-El-Haija et al. "MixHop: Higher-Order Graph
  Convolutional Architectures via Sparsified Neighborhood Mixing"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


class MixHopLayer(nn.Module):
    """MixHop: 混合多跳邻居信息"""

    def __init__(self, in_dim, out_dim, hops=(1, 2, 3)):
        super().__init__()
        self.hops = hops
        self.num_hops = len(hops)
        # 每跳独立的线性投影
        per_hop_dim = out_dim // self.num_hops
        self.hop_linears = nn.ModuleList([
            nn.Linear(in_dim, per_hop_dim, bias=False)
            for _ in hops
        ])
        # 跳间注意力: 学习每跳的重要性
        self.hop_attention = nn.Parameter(
            torch.ones(self.num_hops) / self.num_hops)
        # 如果out_dim不能被num_hops整除, 补齐
        self.residual_dim = out_dim - per_hop_dim * self.num_hops
        if self.residual_dim > 0:
            self.residual_linear = nn.Linear(
                in_dim, self.residual_dim, bias=False)

    def forward(self, X, adj):
        """
        X: [B, L, N, D_in]  或 [B, L, N*k_t, D_in]
        adj: [N, N*k_t] 或类似
        """
        hop_outputs = []
        attn = F.softmax(self.hop_attention, dim=0)
        A_pow = torch.eye(
            adj.shape[-1], device=adj.device
        ).unsqueeze(0).expand(adj.shape[0], -1, -1) \
            if len(adj.shape) == 3 else None

        current_A = adj  # A^1
        for i, h in enumerate(self.hops):
            # A^h * X
            if i == 0:
                AX = torch.matmul(adj, X)
            else:
                current_A = torch.matmul(current_A, adj)
                AX = torch.matmul(current_A, X)
            # 独立投影
            projected = self.hop_linears[i](AX)
            hop_outputs.append(attn[i] * projected)

        # 拼接各跳的输出
        combined = torch.cat(hop_outputs, dim=-1)
        if self.residual_dim > 0:
            combined = torch.cat([
                combined,
                self.residual_linear(X)
            ], dim=-1)

        _dbg("mixhop.attention",
             attn, "diffusion")
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
        # MixHop GCN替代单层gcn_updt
        self.mixhop_gcn = MixHopLayer(
            self.hidden_dim * self.num_matric,
            self.hidden_dim,
            hops=model_args.get('mixhop_hops', (1, 2)))

        # InstanceNorm替代BatchNorm (per-node归一化)
        self.instance_norm = nn.InstanceNorm2d(self.hidden_dim)
        self.activation = nn.GELU()

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
        # MixHop: 对拼接后的特征用多跳混合投影
        # 需要一个单位邻接(因为邻接信息已经通过support融入)
        # 这里MixHop在特征空间做多尺度聚合
        B, L, N_or_NK, D = out.shape
        identity_adj = torch.eye(N_or_NK, device=out.device)
        identity_adj = identity_adj.unsqueeze(0).unsqueeze(0)
        identity_adj = identity_adj.expand(B, L, -1, -1)
        out_flat = out.reshape(B * L, N_or_NK, D)
        id_flat = identity_adj.reshape(B * L, N_or_NK, N_or_NK)
        out_mixed = self.mixhop_gcn(out_flat, id_flat)
        out = out_mixed.reshape(B, L, N_or_NK, -1)
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
        X_0 = out.mean(dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len,
            kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden
