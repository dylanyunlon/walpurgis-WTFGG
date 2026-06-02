"""
Spatially-Temporally Localized Convolution for the diffusion branch.
"""
import sys
import torch
import torch.nn as nn

_DBG = ("--debug-stconv" in sys.argv)


class STLocalizedConv(nn.Module):
    """Graph conv that fuses temporal and spatial kernels into a localized operator."""

    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None, sta_graph=None, **kw):
        super().__init__()
        self.k_s = kw['k_s']
        self.k_t = kw['k_t']
        self.hidden_dim = hidden_dim

        # graph switches
        self.pre_graphs = pre_defined_graph
        self.use_pre  = use_pre
        self.use_dyn  = dy_graph
        self.use_sta  = sta_graph

        self.support_len = len(self.pre_graphs) + int(dy_graph) + int(sta_graph)
        self.n_matrices = (
            int(use_pre) * len(self.pre_graphs)
            + len(self.pre_graphs) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1

        self.drop = nn.Dropout(kw['dropout'])
        self.pre_graphs = self._expand_predef(self.pre_graphs)

        self.fc_update = nn.Linear(self.k_t * hidden_dim,
                                   self.k_t * hidden_dim, bias=False)
        self.gcn_agg   = nn.Linear(hidden_dim * self.n_matrices, hidden_dim)

        self.bn   = nn.BatchNorm2d(hidden_dim)
        self.act  = nn.ReLU()

    # ---- graph expansion ----

    def _expand_predef(self, graphs):
        """Pre-compute multi-order + ST-localized form of static graphs."""
        expanded = []
        eye = 1 - torch.eye(graphs[0].shape[0]).to(graphs[0].device)
        for g in graphs:
            power = g
            expanded.append(power * eye)
            for _k in range(2, self.k_s + 1):
                power = torch.matmul(g, power)
                expanded.append(power * eye)
        # reshape each to (N, k_t*N) for localized convolution
        st = []
        for g in expanded:
            g2 = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            g2 = g2.reshape(g2.shape[0], g2.shape[1] * g2.shape[2])
            st.append(g2)
        return st

    def gconv(self, support, X_k, X_0):
        parts = [X_0]
        for g in support:
            if g.dim() == 2:
                pass
            else:
                g = g.unsqueeze(1)
            parts.append(torch.matmul(g, X_k))
        merged = torch.cat(parts, dim=-1)
        out = self.gcn_agg(merged)
        out = self.drop(out)
        if _DBG:
            print(f"[DBG:stconv] gconv  "
                  f"n_support={len(support)}  "
                  f"out.shape={tuple(out.shape)}  "
                  f"out_mean={out.mean().item():.4f}")
        return out

    def forward(self, X, dynamic_graph, static_graph):
        """X: (B, L, N, D) -> (B, L', N, D)."""
        # unfold temporal kernel
        X_u = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, L, N, K, D = X_u.shape

        if _DBG:
            print(f"[DBG:stconv] forward  input={tuple(X.shape)}  "
                  f"unfolded=({B},{L},{N},{K},{D})")

        # assemble support set
        support = []
        if self.use_pre:
            support += self.pre_graphs
        if self.use_dyn:
            support += dynamic_graph
        if self.use_sta:
            support += self._expand_predef(static_graph)

        # parallelize over temporal dim
        flat = X_u.reshape(B, L, N, K * D)
        out  = self.act(self.fc_update(flat))
        out  = out.view(B, L, N, K, D)

        X_0 = out.mean(dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, L, K * N, D)

        hidden = self.gconv(support, X_k, X_0)
        return hidden
