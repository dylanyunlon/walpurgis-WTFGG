"""Flux STLocalizedConv: 流式感知的时空局部卷积.
与upstream(标准unfold+gconv)和vortex(同upstream)不同,
Flux在时序展开后施加因果掩码: 未来时间步的graph conv贡献被置零,
确保流式推理时不会leak未来信息."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None,
                 sta_graph=None, **model_args):
        super().__init__()
        # gated temporal conv
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim
        # graph conv
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph
        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (
            int(use_pre) * len(self.pre_defined_graph) +
            len(self.pre_defined_graph) * int(dy_graph) +
            int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(
            self.pre_defined_graph)
        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim,
            self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric,
            self.hidden_dim)
        # others
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()
        # Flux: 因果时序掩码权重 — 可学习
        self.causal_mask_scale = nn.Parameter(
            torch.tensor(1.0))

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

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(
            support[0].shape[0]).to(support[0].device)
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
        # X: [bs, seq, nodes, feat]
        X = X.unfold(1, self.k_t, 1).permute(
            0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, \
            num_feat = X.shape
        # support
        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)
        # parallelize
        X = X.reshape(
            batch_size, seq_len, num_nodes,
            kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(
            batch_size, seq_len, num_nodes,
            kernel_size, num_feat)
        # Flux: 因果时序掩码 — 对kernel内的时间步施加递减权重
        causal_scale = torch.sigmoid(
            self.causal_mask_scale)
        causal_weights = torch.linspace(
            causal_scale.item() * 0.5, 1.0, kernel_size,
            device=out.device)
        causal_weights = causal_weights.view(
            1, 1, 1, kernel_size, 1)
        out = out * causal_weights
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len,
            kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        if _FX_DBG:
            print(f"[FX:st_conv] seq_len={seq_len} "
                  f"causal_scale={causal_scale.item():.4f} "
                  f"hidden_norm={hidden.norm().item():.4f}",
                  file=sys.stderr)
        return hidden
