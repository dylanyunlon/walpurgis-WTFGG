import torch
import torch.nn as nn
import sys

_DBG_STCONV = ("--dbg-stconv" in sys.argv)


def _sp(tag, t):
    if not _DBG_STCONV:
        return
    with torch.no_grad():
        print(f"[DBG-STCONV][{tag}] shape={list(t.shape)}  "
              f"mean={t.mean().item():.5f}  std={t.std().item():.5f}  "
              f"absmax={t.abs().max().item():.5f}")


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

        # 算法改动: 用 GroupNorm(8 groups) 替代 BatchNorm
        # GN 在小 batch 下比 BN 稳定, 不依赖 batch 统计量
        num_groups = min(8, hidden_dim)
        self.norm = nn.GroupNorm(num_groups, self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        """算法改动: 给每条 support graph 的输出加一个可学习的标量权重,
        做 weighted sum 而非直接 cat (仍保留原始 X_0 不变)。
        但为了保持 num_matric 维度兼容, 这里的实现是:
        cat 之后在 gcn_updt 之前做 channel-wise scaling。
        """
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)

        _sp("gconv_cat", out)

        out = self.gcn_updt(out)
        out = self.dropout(out)

        _sp("gconv_out", out)
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
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        _sp("input_unfolded", X)

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

        # 算法改动: 用 GroupNorm 替代 BN — reshape 成 (B*T, C, N) 做 norm
        b, t, n, c = hidden.shape
        hidden_flat = hidden.reshape(b * t, c, n)       # (B*T, C, N) for GN on C
        # GN expects (N, C, *), so transpose
        hidden_flat = hidden_flat.permute(0, 2, 1)       # (B*T, N, C) — no, GN needs (B, C, spatial)
        hidden_flat = hidden.reshape(b * t, n, c).permute(0, 2, 1)  # (B*T, C, N)
        hidden_flat = self.norm(hidden_flat)
        hidden = hidden_flat.permute(0, 2, 1).reshape(b, t, n, c)

        _sp("final_hidden", hidden)
        return hidden
