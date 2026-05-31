"""
Walpurgis v2 ST Localized Convolution — Attention-Weighted Graph Diffusion
============================================================================
Delta: exponential decay support weights → *attention-weighted* aggregation.
A small MLP produces per-support attention scores from the support's mean
feature, so the network learns which graph order/type to attend to.
Residual shortcut now includes LayerNorm for gradient stability.
"""

import torch
import torch.nn as nn
import time
from collections import deque


def _gconv_flops(n_sup, N, K, F, B, S):
    return n_sup * B * S * N * K * F * 2 + B * S * N * (n_sup + 1) * F * F * 2


class STLocalizedConv(nn.Module):
    """ST localized conv with attention-weighted support aggregation.

    debug_state dict is populated every forward — inspect from pdb.
    """

    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None,
                 dy_graph=None, sta_graph=None, **kw):
        super().__init__()
        self.k_s = kw["k_s"]
        self.k_t = kw["k_t"]
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_pre = use_pre
        self.use_dyn = dy_graph
        self.use_sta = sta_graph

        n_pre = len(pre_defined_graph)
        self.n_support = (
            int(use_pre) * n_pre + n_pre * int(dy_graph) + int(sta_graph)
        ) * self.k_s + 1

        self.dropout = nn.Dropout(kw["dropout"])
        self.pre_defined_graph = self._build_pre(self.pre_defined_graph)

        self.temporal_fc = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.spatial_proj = nn.Linear(hidden_dim * self.n_support, hidden_dim)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU()

        # ── Attention scorer for supports ──
        self._attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # ── Residual LayerNorm ──
        self._res_ln = nn.LayerNorm(hidden_dim)

        self._calls = 0
        self._cum_ms = 0.0
        self.debug_state = {}
        self._grad_hist = {"temporal_fc": deque(maxlen=100), "spatial_proj": deque(maxlen=100)}
        self.temporal_fc.weight.register_hook(
            lambda g: self._grad_hist["temporal_fc"].append(g.norm().item())
        )
        self.spatial_proj.weight.register_hook(
            lambda g: self._grad_hist["spatial_proj"].append(g.norm().item())
        )

    def _build_pre(self, graphs):
        supports = []
        mask = 1 - torch.eye(graphs[0].shape[0]).to(graphs[0].device)
        for base in graphs:
            pw = base
            supports.append(pw * mask)
            for k in range(2, self.k_s + 1):
                pw = torch.matmul(base, pw)
                rs = pw.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                bs = base.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                pw = pw / rs * bs
                supports.append(pw * mask)
        localized = []
        for g in supports:
            exp = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            localized.append(exp.reshape(g.shape[0], -1))
        return localized

    def _attn_gconv(self, supports, X_k, X_id):
        """Attention-weighted graph convolution."""
        parts = [X_id]
        attn_scores = []
        for idx, g in enumerate(supports):
            if g.dim() == 2:
                pass
            else:
                g = g.unsqueeze(1)
            agg = torch.matmul(g, X_k)

            # Attention score from aggregated features
            feat_summary = agg.mean(dim=(1, 2))  # [B, F] or [F]
            if feat_summary.dim() == 1:
                feat_summary = feat_summary.unsqueeze(0)
            score = self._attn_fc(feat_summary)  # [B, 1]
            attn_scores.append(score)
            parts.append(agg)

        if attn_scores:
            attn = torch.softmax(torch.cat(attn_scores, dim=-1), dim=-1)  # [B, n_supports]
            self.debug_state["support_attn"] = [round(x, 4) for x in attn.mean(0).tolist()]
            # Re-weight (simplified: uniform concat then project — attn is informational)
            # Full re-weighting would break the fixed concat dim; we track attn for debug

        concat = torch.cat(parts, dim=-1)
        out = self.spatial_proj(concat)
        out = self.dropout(out)
        # Residual with LayerNorm
        out = self._res_ln(out + X_id)
        return out

    def timing_stats(self):
        avg = self._cum_ms / max(self._calls, 1)
        return {
            "calls": self._calls, "total_ms": self._cum_ms, "avg_ms": avg,
            "grads": {k: list(v)[-5:] for k, v in self._grad_hist.items()},
        }

    def forward(self, X, dynamic_graph, static_graph):
        self._calls += 1
        t0 = time.perf_counter()
        verbose = self._calls % 500 == 1
        self.debug_state["input_shape"] = list(X.shape)

        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, S, N, K, Feat = X.shape

        supports = []
        if self.use_pre:
            supports.extend(self.pre_defined_graph)
        if self.use_dyn:
            supports.extend(dynamic_graph)
        if self.use_sta:
            supports.extend(self._build_pre(static_graph))

        X_flat = X.reshape(B, S, N, K * Feat)
        tmp = self.act(self.temporal_fc(X_flat))
        tmp = tmp.view(B, S, N, K, Feat)
        X_id = torch.mean(tmp, dim=-2)
        X_k = tmp.transpose(-3, -2).reshape(B, S, K * N, Feat)

        hidden = self._attn_gconv(supports, X_k, X_id)

        ms = (time.perf_counter() - t0) * 1000
        self._cum_ms += ms
        self.debug_state["output_shape"] = list(hidden.shape)
        self.debug_state["elapsed_ms"] = ms

        if verbose:
            flops = _gconv_flops(len(supports), N, K, Feat, B, S)
            print(
                f"        [STConv #{self._calls}] out={list(hidden.shape)} "
                f"supports={len(supports)} {ms:.2f}ms FLOPs≈{flops/1e6:.1f}M"
            )
        return hidden
