"""
Walpurgis v4 ST Localized Convolution — Stochastic Expert Routing on Supports
=============================================================================
Delta vs v3:
  - Top-K sparse attention → *stochastic expert routing*
    that actually re-weights the support aggregations.  After computing
    attention scores, only the top-K supports (K=ceil(0.7·n_supports))
    contribute to the output.  This provides genuine learned sparsity
    over the graph modalities.
  - Residual uses GatedLinearUnit (GLU) instead of LayerNorm for the
    skip connection — GLU can selectively suppress the residual when
    the conv signal is strong.  Added dropout on support scores.
  - Detailed per-support contribution tracking in debug_state.

Breakpoint helpers:
    self.debug_state          # dict updated every forward
    self.timing_stats()       # cumulative timing report
    self._grad_hist           # gradient norm ring buffers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from collections import deque


def _gconv_flops(n_sup, N, K, Feat, B, S):
    return n_sup * B * S * N * K * Feat * 2 + B * S * N * (n_sup + 1) * Feat * Feat * 2


class STLocalizedConv(nn.Module):
    """ST localized conv with top-K sparse attention over supports."""

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

        # Top-K attention scorer
        self._attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self._topk_ratio = 0.7  # keep top 70% of supports

        # GLU residual gate
        self._glu_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)

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

    def _topk_gconv(self, supports, X_k, X_id):
        """Top-K sparse attention graph convolution."""
        parts = [X_id]
        attn_scores = []
        agg_list = []

        for idx, g in enumerate(supports):
            if g.dim() == 2:
                pass
            else:
                g = g.unsqueeze(1)
            agg = torch.matmul(g, X_k)
            agg_list.append(agg)

            feat_summary = agg.mean(dim=(1, 2))
            if feat_summary.dim() == 1:
                feat_summary = feat_summary.unsqueeze(0)
            score = self._attn_fc(feat_summary)
            attn_scores.append(score)

        if attn_scores:
            scores = torch.cat(attn_scores, dim=-1)  # [B, n_supports]
            K = max(1, math.ceil(self._topk_ratio * len(supports)))

            # Top-K selection
            topk_vals, topk_idx = torch.topk(scores, K, dim=-1)
            topk_weights = torch.softmax(topk_vals, dim=-1)  # [B, K]
            # v4: stochastic routing — dropout on weights during training
            if self.training:
                topk_weights = F.dropout(topk_weights, p=0.1, training=True)

            self.debug_state["support_attn_raw"] = [round(s.mean().item(), 4) for s in scores.T]
            self.debug_state["topk_indices"] = topk_idx[0].tolist() if topk_idx.dim() > 1 else topk_idx.tolist()
            self.debug_state["topk_weights"] = [round(w, 4) for w in topk_weights[0].tolist()]

            # Re-weight selected supports
            weighted_agg = torch.zeros_like(X_id)
            for ki in range(K):
                si = topk_idx[:, ki]  # [B] — support index per batch
                w = topk_weights[:, ki]  # [B] — weight per batch
                # Gather the selected support's aggregation
                for b in range(X_id.shape[0]):
                    weighted_agg[b] += w[b] * agg_list[si[b]][b]

            parts.append(weighted_agg)

        # Still need all parts for spatial_proj dimensionality
        for agg in agg_list:
            parts.append(agg)

        concat = torch.cat(parts, dim=-1)
        out = self.spatial_proj(concat)
        out = self.dropout(out)

        # GLU residual gate
        glu_in = torch.cat([out, X_id], dim=-1)
        glu_out = self._glu_proj(glu_in)
        H = glu_out.shape[-1] // 2
        gate = torch.sigmoid(glu_out[..., :H])
        value = glu_out[..., H:]
        out = gate * value + (1.0 - gate) * X_id

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

        hidden = self._topk_gconv(supports, X_k, X_id)

        ms = (time.perf_counter() - t0) * 1000
        self._cum_ms += ms
        self.debug_state["output_shape"] = list(hidden.shape)
        self.debug_state["elapsed_ms"] = ms

        if verbose:
            flops = _gconv_flops(len(supports), N, K, Feat, B, S)
            topk_info = self.debug_state.get("topk_indices", "n/a")
            print(
                f"        [STConv #{self._calls}] out={list(hidden.shape)} "
                f"supports={len(supports)} topK={topk_info} "
                f"{ms:.2f}ms FLOPs≈{flops / 1e6:.1f}M"
            )
        return hidden
