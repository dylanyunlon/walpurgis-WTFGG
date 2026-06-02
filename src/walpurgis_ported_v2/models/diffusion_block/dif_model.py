"""
Spatial-Temporal Localized Convolution (STLocalizedConv).
Performs graph convolution over a sliding temporal kernel window,
combining predefined, dynamic, and static hidden graphs.
"""

import torch
import torch.nn as nn
import sys

_DBG_STCONV = ("--debug-stconv" in sys.argv) or False


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None,
                 dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']           # spatial diffusion order
        self.k_t = model_args['k_t']           # temporal kernel size
        self.d_hidden = hidden_dim

        # graph configuration flags
        self.predef_graphs = pre_defined_graph
        self.use_predef   = use_pre
        self.use_dynamic  = dy_graph
        self.use_static   = sta_graph

        # count how many adjacency matrices will be stacked
        n_predef = len(self.predef_graphs)
        self.n_support = n_predef + int(dy_graph) + int(sta_graph)
        self.n_matrices = (
            int(use_pre) * n_predef +
            n_predef * int(dy_graph) +
            int(sta_graph)
        ) * self.k_s + 1                       # +1 for identity / X_0

        self.dropout = nn.Dropout(model_args['dropout'])
        # preprocess static predefined graphs once
        self.predef_graphs = self._preprocess_static(self.predef_graphs)

        # temporal fusion FC (operates on unfolded kernel dimension)
        self.temporal_fc = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False
        )
        # graph conv output projection
        self.gcn_proj = nn.Linear(hidden_dim * self.n_matrices, hidden_dim)

        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU()

    # ─── graph conv ───

    def _gconv(self, support, X_k, X_0):
        """
        Aggregate over all support graphs.
        X_0: [B, L, N, D]   — mean-pooled node features (identity term)
        X_k: [B, L, k_t*N, D] — kernel-expanded features for each graph
        """
        parts = [X_0]
        for g in support:
            if len(g.shape) == 2:       # static / predefined: [N, k_t*N]
                h = torch.matmul(g, X_k)
            else:                       # dynamic: [B, N, k_t*N]
                h = torch.matmul(g.unsqueeze(1), X_k)
            parts.append(h)
        out = torch.cat(parts, dim=-1)
        out = self.gcn_proj(out)
        out = self.dropout(out)
        return out

    # ─── static graph preprocessing ───

    def _preprocess_static(self, graphs):
        """Expand static graphs to k-order, then tile for temporal kernel."""
        ordered = []
        eye_mask = 1 - torch.eye(graphs[0].shape[0]).to(graphs[0].device)
        for g in graphs:
            power = g
            ordered.append(power * eye_mask)
            for _ in range(2, self.k_s + 1):
                power = torch.matmul(g, power)
                ordered.append(power * eye_mask)
        # tile each order graph along the temporal kernel dimension
        st_graphs = []
        for g in ordered:
            tiled = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            tiled = tiled.reshape(tiled.shape[0], tiled.shape[1] * tiled.shape[2])
            st_graphs.append(tiled)
        return st_graphs

    # ─── forward ───

    def forward(self, X, dynamic_graph, static_graph):
        """
        X: [B, L, N, D]
        Returns: [B, L', N, D]  where L' = L - k_t + 1
        """
        # unfold temporal kernel: [B, L', N, k_t, D]
        X_unfold = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, L_out, N, K, D = X_unfold.shape

        if _DBG_STCONV:
            print(f"[DBG:stconv] forward  input={tuple(X.shape)}  "
                  f"unfolded={tuple(X_unfold.shape)}")

        # assemble support set
        support = []
        if self.use_predef:
            support += self.predef_graphs
        if self.use_dynamic:
            support += dynamic_graph
        if self.use_static:
            support += self._preprocess_static(static_graph)

        # flatten kernel into feature dim: [B, L', N, k_t * D]
        X_flat = X_unfold.reshape(B, L_out, N, K * D)
        # temporal fusion
        fused = self.act(self.temporal_fc(X_flat))
        # restore kernel view: [B, L', N, k_t, D]
        fused = fused.view(B, L_out, N, K, D)
        # X_0 = mean over kernel (identity-like term)
        X_0 = torch.mean(fused, dim=-2)
        # X_k = reshape for graph matmul: [B, L', k_t*N, D]
        X_k = fused.transpose(-3, -2).reshape(B, L_out, K * N, D)

        hidden = self._gconv(support, X_k, X_0)

        if _DBG_STCONV:
            print(f"[DBG:stconv] output  shape={tuple(hidden.shape)}  "
                  f"norm={hidden.norm().item():.4f}  "
                  f"n_support={len(support)}")
        return hidden
