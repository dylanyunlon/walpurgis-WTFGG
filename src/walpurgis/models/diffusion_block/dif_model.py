import torch
import torch.nn as nn
from walpurgis import _dbg

_TAG = "stconv"


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
        self.pre_defined_graph = self._preprocess_graph(self.pre_defined_graph)

        # 改动4: 单层 FC → 两层 FC + GELU
        # upstream: 1个 Linear(k_t*D, k_t*D)
        # walpurgis改动: Linear → GELU → Linear, 增加非线性容量
        kt_dim = self.k_t * hidden_dim
        self.fc_pre = nn.Linear(kt_dim, kt_dim, bias=False)
        self.fc_post = nn.Linear(kt_dim, kt_dim, bias=False)

        self.gcn_updt = nn.Linear(hidden_dim * self.num_matric, hidden_dim)

        # 改动1: BN → InstanceNorm2d
        # InstanceNorm 对每个 sample 的每个 channel 独立归一化
        # 比 BN 更适合小 batch + 变长序列
        self.norm = nn.InstanceNorm2d(hidden_dim, affine=True)

        # 改动2: ReLU → GELU
        self.act = nn.GELU()

        # 改动3: gconv skip projection — 让 X_0 能直接加到输出
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

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

        # 改动3: 残差 skip — upstream 无此连接
        out = out + self.skip_proj(X_0)

        _dbg(_TAG, "gconv_out", out=out, num_support=len(support))
        return out

    def _preprocess_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            # 改动5: fill_diagonal_ 原地清零对角线
            cleaned = k_1_order * mask
            cleaned.fill_diagonal_(0)
            graph_ordered.append(cleaned)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                cleaned_k = k_1_order * mask
                cleaned_k.fill_diagonal_(0)
                graph_ordered.append(cleaned_k)
        st_local_graph = []
        for g in graph_ordered:
            g_exp = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            g_exp = g_exp.reshape(g_exp.shape[0], g_exp.shape[1] * g_exp.shape[2])
            st_local_graph.append(g_exp)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, L, N, K, D = X.shape

        _dbg(_TAG, "unfold", X=X)

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self._preprocess_graph(static_graph)

        X = X.reshape(B, L, N, K * D)

        # 改动4: 两层 FC + GELU 而非单层
        out = self.fc_pre(X)
        out = self.act(out)
        out = self.fc_post(out)
        out = self.act(out)

        out = out.view(B, L, N, K, D)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, L, K * N, D)

        hidden = self.gconv(support, X_k, X_0)

        # 改动1: InstanceNorm — (B, L, N, D) → (B*L, D, N, 1) → IN → 还原
        h_shape = hidden.shape
        h_in = hidden.reshape(-1, h_shape[-1], h_shape[-2], 1)  # (B*L, D, N, 1)
        h_in = self.norm(h_in)
        hidden = h_in.reshape(h_shape)

        _dbg(_TAG, "forward_out", hidden=hidden)
        return hidden
