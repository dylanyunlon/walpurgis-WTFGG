"""Vortex STLocalizedConv: SpectralNorm + LeakyReLU + Chebyshev polynomial aggregation.
Unlike upstream (BatchNorm+ReLU, simple k-power graph) and eclipse (InstanceNorm+GELU+skip),
Vortex uses spectral normalization for training stability, LeakyReLU for avoiding dead
neurons, and Chebyshev polynomial-based graph aggregation for capturing richer spectral
patterns in the graph convolution."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
from torch.nn.utils import spectral_norm
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']; self.k_t = model_args['k_t']; self.hidden_dim = hidden_dim
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre; self.use_dynamic_hidden_graph = dy_graph; self.use_static__hidden_graph = sta_graph
        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self._chebyshev_graph(self.pre_defined_graph)
        # SpectralNorm on linear layers for Lipschitz-bounded training
        self.fc_list_updt = spectral_norm(nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False))
        self.gcn_updt = spectral_norm(nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim))
        # LeakyReLU for non-zero gradients everywhere (vs upstream ReLU, eclipse GELU)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # Chebyshev polynomial coefficients (learnable)
        self.cheb_coeffs = nn.Parameter(torch.ones(self.k_s + 1) / (self.k_s + 1))

    def _chebyshev_graph(self, support):
        """Chebyshev polynomial-based graph expansion.
        T_0(L) = I, T_1(L) = L, T_k(L) = 2*L*T_{k-1} - T_{k-2}
        This captures richer spectral information than simple k-th powers."""
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            # T_0 = I (identity, added as self-loop in num_matric calculation)
            t_prev = torch.eye(graph.shape[0]).to(graph.device)
            t_curr = graph
            graph_ordered.append(t_curr * mask)  # T_1
            for k in range(2, self.k_s + 1):
                # Chebyshev recurrence: T_k = 2*G*T_{k-1} - T_{k-2}
                t_next = 2.0 * torch.matmul(graph, t_curr) - t_prev
                graph_ordered.append(t_next * mask)
                t_prev = t_curr
                t_curr = t_next
        # ST localization
        st_local = []
        for g in graph_ordered:
            g = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            st_local.append(g.reshape(g.shape[0], g.shape[1] * g.shape[2]))
        return st_local

    def gconv(self, support, X_k, X_0):
        """Graph convolution with Chebyshev-weighted aggregation."""
        out = [X_0]
        for i, graph in enumerate(support):
            if len(graph.shape) != 2: graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            # Weight by learnable Chebyshev coefficients
            coeff_idx = min(i, len(self.cheb_coeffs) - 1)
            coeff = torch.softmax(self.cheb_coeffs, dim=0)[coeff_idx]
            out.append(H_k * coeff)
        out = self.gcn_updt(torch.cat(out, dim=-1))
        out = self.dropout(out)
        if _VX_DBG:
            print(f"[VX:gconv@dif_model] out={list(out.shape)} cheb_coeffs={torch.softmax(self.cheb_coeffs,dim=0).detach().tolist()}", file=sys.stderr)
        return out

    def get_graph(self, support):
        """For static/dynamic graphs at runtime (not using Chebyshev expansion)."""
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            t_prev = torch.eye(graph.shape[0]).to(graph.device)
            t_curr = graph
            graph_ordered.append(t_curr * mask)
            for k in range(2, self.k_s + 1):
                t_next = 2.0 * torch.matmul(graph, t_curr) - t_prev
                graph_ordered.append(t_next * mask)
                t_prev = t_curr; t_curr = t_next
        st_local = []
        for g in graph_ordered:
            g = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            st_local.append(g.reshape(g.shape[0], g.shape[1] * g.shape[2]))
        return st_local

    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, S, N, K, Feat = X.shape
        support = []
        if self.use_predefined_graph: support += self.pre_defined_graph
        if self.use_dynamic_hidden_graph: support += dynamic_graph
        if self.use_static__hidden_graph: support += self.get_graph(static_graph)
        X = X.reshape(B, S, N, K * Feat)
        out = self.activation(self.fc_list_updt(X))
        out = out.view(B, S, N, K, Feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(B, S, K * N, Feat)
        hidden = self.gconv(support, X_k, X_0)
        if _VX_DBG:
            print(f"[VX:forward@dif_model] hidden={list(hidden.shape)} range=[{hidden.min().item():.4f},{hidden.max().item():.4f}]", file=sys.stderr)
        return hidden
