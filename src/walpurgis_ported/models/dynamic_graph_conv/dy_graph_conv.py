"""
Walpurgis v2 Dynamic Graph Constructor — Adaptive Sparse Graph Learning
=========================================================================
Re-ported with ≈20 % algorithmic delta.

Deltas:
  1. Density floor: fixed threshold → *EMA-adaptive* threshold.
     The floor tracks recent graph density via EMA and only triggers
     rescue when density drops below 0.6× the running average.
  2. Graph cache: fingerprint-based hash → *cosine similarity* between
     flattened graph vectors.  Reuse when cosine > 0.998.
  3. Spectral clamp: fixed ceiling → *percentile-based* ceiling
     (p99 of row-sums), preventing unnecessary clamping on dense graphs.
  4. Added .sparsity_log — a deque of (call_n, density, rescued) tuples
     for post-mortem sparsity trend analysis.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
from collections import deque

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_RESCUE_EMA_COEFF = 0.05
_RESCUE_RATIO = 0.6           # rescue if density < ema * ratio
_RESCUE_ALPHA_CAP = 0.3
_CACHE_COSINE_THRESH = 0.998
_SPECTRAL_PCTL = 99           # row-sum percentile for clamping


class DynamicGraphConstructor(nn.Module):
    """Learns dynamic spatial graphs with EMA-adaptive density rescue.

    Pipeline:  distance → mask → normalize → density_rescue → multi_order
              → spectral_clamp → st_localize → cache_check

    Debug helpers (call from pdb):
        self.stage_profile()          # timing percentiles per stage
        self.graph_quality_report()   # current graph stats
        self.sparsity_trend(20)       # last 20 density readings
    """

    _global_n = 0

    def __init__(self, **kw):
        super().__init__()
        self.k_s = kw["k_s"]
        self.k_t = kw["k_t"]
        self.hidden_dim = kw["num_hidden"]
        self.node_dim = kw["node_hidden"]

        self.distance_fn = DistanceFunction(**kw)
        self.mask_fn = Mask(**kw)
        self.norm_fn = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        # ── EMA density tracking ──
        self._ema_density = 0.3      # initial guess
        # ── Cosine cache ──
        self._prev_flat = None
        self._prev_graphs = None
        self._hits = 0
        self._misses = 0
        # ── Stage timing ──
        self._timers = {
            s: deque(maxlen=500)
            for s in [
                "distance", "mask", "normalize", "rescue",
                "multi_order", "spectral_clamp", "st_local",
            ]
        }
        # ── Sparsity log ──
        self.sparsity_log = deque(maxlen=2000)

        tp = sum(p.numel() for p in self.parameters())
        print(
            f"[DynGraph] k_s={self.k_s} k_t={self.k_t} "
            f"hidden={self.hidden_dim} node={self.node_dim} params={tp:,}"
        )

    def _inspect(self, tag, t, show):
        if not show or t is None:
            return
        with torch.no_grad():
            tf = t.detach().float()
            dens = (tf.abs() > 1e-6).float().mean().item()
            fl = ""
            if torch.isnan(tf).any():
                fl += " \033[91mNaN\033[0m"
            if torch.isinf(tf).any():
                fl += " \033[91mInf\033[0m"
            print(
                f"    [{tag}] shape={list(t.shape)} "
                f"μ={tf.mean().item():.5f} σ={tf.std().item():.5f} "
                f"∈[{tf.min().item():.5f},{tf.max().item():.5f}] "
                f"dens={dens:.4f}{fl}"
            )

    def _rescue_sparse(self, adj, show):
        """EMA-adaptive density floor rescue."""
        with torch.no_grad():
            dens = (adj.abs() > 1e-6).float().mean().item()

        # Update EMA
        self._ema_density = _RESCUE_EMA_COEFF * dens + (1 - _RESCUE_EMA_COEFF) * self._ema_density
        threshold = self._ema_density * _RESCUE_RATIO

        self.sparsity_log.append((DynamicGraphConstructor._global_n, dens, False))

        if dens >= threshold:
            if show:
                print(f"    [density] {dens:.4f} ≥ thresh {threshold:.4f} (ema={self._ema_density:.4f}) → OK")
            return adj, False

        shortfall = (threshold - dens) / max(threshold, 1e-8)
        alpha = min(shortfall, _RESCUE_ALPHA_CAP)
        B, N = adj.shape[0], adj.shape[1]
        scale = adj.abs().mean().item() + 1e-8
        diag = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
        adj = adj + alpha * scale * diag

        self.sparsity_log[-1] = (DynamicGraphConstructor._global_n, dens, True)

        if show:
            new_d = (adj.abs() > 1e-6).float().mean().item()
            print(
                f"    [density] RESCUE: {dens:.4f} → {new_d:.4f} "
                f"(α={alpha:.4f} thresh={threshold:.4f} ema={self._ema_density:.4f})"
            )
        return adj, True

    def _spectral_clamp(self, ordered, show):
        """Percentile-based spectral clamping on high-order powers."""
        result = []
        for mi, modality in enumerate(ordered):
            clamped = []
            for ki, g in enumerate(modality):
                if ki >= 2:
                    rs = g.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    ceiling = torch.quantile(rs.float(), _SPECTRAL_PCTL / 100.0).item()
                    peak = rs.max().item()
                    if peak > ceiling * 1.5:
                        g = g / rs * ceiling
                        if show:
                            print(
                                f"    [spectral] mod={mi} k={ki} "
                                f"peak={peak:.2f} p{_SPECTRAL_PCTL}={ceiling:.2f} → clamped"
                            )
                clamped.append(g)
            result.append(clamped)
        return result

    def _cosine_cache_check(self, adj):
        """Check cache via cosine similarity on flattened graph vectors."""
        with torch.no_grad():
            flat = adj.detach().float().view(-1)
            if self._prev_flat is not None:
                cos = nnF.cosine_similarity(flat.unsqueeze(0), self._prev_flat.unsqueeze(0)).item()
                if cos >= _CACHE_COSINE_THRESH and self._prev_graphs is not None:
                    self._hits += 1
                    return True, cos
            self._prev_flat = flat.clone()
            self._misses += 1
            return False, 0.0

    def _localize_st(self, ordered):
        localized = []
        for mod in ordered:
            for g in mod:
                expanded = g.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                flat = expanded.reshape(expanded.shape[0], expanded.shape[1], -1)
                localized.append(flat)
        return localized

    def sparsity_trend(self, n=20):
        """Print recent sparsity readings — call from pdb."""
        entries = list(self.sparsity_log)[-n:]
        print(f"  [Sparsity Trend] last {len(entries)} entries:")
        for call_n, dens, rescued in entries:
            tag = " RESCUED" if rescued else ""
            print(f"    #{call_n}: density={dens:.4f}{tag}")

    def stage_profile(self):
        print(f"\n  [DynGraph Pipeline @ call {DynamicGraphConstructor._global_n}]")
        for st, buf in self._timers.items():
            if buf:
                arr = np.array(buf)
                print(f"    {st:>16s}: μ={arr.mean():.3f}ms  p50={np.median(arr):.3f}ms  p99={np.percentile(arr,99):.3f}ms")
        total = self._hits + self._misses
        if total:
            print(f"    cache: {self._hits}/{total} hits ({self._hits/total*100:.1f}%)")

    def graph_quality_report(self):
        if self._prev_graphs is None:
            print("  [Graph Quality] no graphs yet")
            return
        for i, g in enumerate(self._prev_graphs):
            dens = (g.abs() > 1e-6).float().mean().item()
            print(f"  graph[{i}]: shape={list(g.shape)} density={dens:.4f} std={g.std().item():.5f}")

    def forward(self, **inputs):
        DynamicGraphConstructor._global_n += 1
        cn = DynamicGraphConstructor._global_n
        show = cn <= 3 or cn % 200 == 0

        X = inputs["history_data"]
        E_d = inputs["node_embedding_d"]
        E_u = inputs["node_embedding_u"]
        T_D = inputs["time_in_day_feat"]
        D_W = inputs["day_in_week_feat"]

        if show:
            print(f"\n  [DynGraph #{cn}] X={list(X.shape)} E={list(E_d.shape)}")

        # 1. Distance
        t0 = time.perf_counter()
        adj = self.distance_fn(X, E_d, E_u, T_D, D_W)
        self._timers["distance"].append((time.perf_counter() - t0) * 1000)
        if show:
            self._inspect("distance", adj, True)

        # 2. Mask
        t0 = time.perf_counter()
        adj = self.mask_fn(adj)
        self._timers["mask"].append((time.perf_counter() - t0) * 1000)

        # 3. Normalize
        t0 = time.perf_counter()
        adj = self.norm_fn(adj)
        self._timers["normalize"].append((time.perf_counter() - t0) * 1000)

        # 4. EMA density rescue
        t0 = time.perf_counter()
        adj, rescued = self._rescue_sparse(adj, show)
        self._timers["rescue"].append((time.perf_counter() - t0) * 1000)

        # 5. Cosine cache
        hit, cos = self._cosine_cache_check(adj)
        if hit:
            if show:
                print(f"  [CACHE HIT] cos={cos:.5f} reuse={self._hits}/{self._hits+self._misses}")
            return self._prev_graphs

        # 6. Multi-order
        t0 = time.perf_counter()
        ordered = self.multi_order(adj)
        self._timers["multi_order"].append((time.perf_counter() - t0) * 1000)

        # 7. Spectral clamp
        t0 = time.perf_counter()
        ordered = self._spectral_clamp(ordered, show)
        self._timers["spectral_clamp"].append((time.perf_counter() - t0) * 1000)

        # 8. ST localization
        t0 = time.perf_counter()
        graphs = self._localize_st(ordered)
        self._timers["st_local"].append((time.perf_counter() - t0) * 1000)

        self._prev_graphs = graphs

        if show:
            total = sum(self._timers[s][-1] for s in self._timers if self._timers[s])
            tier = "HBM" if total >= 5 else ("GDDR" if total >= 2 else "DRAM")
            print(f"  [Total] {total:.2f}ms → {tier} | graphs={len(graphs)} | rescued={rescued}")

        if cn % 500 == 0:
            self.stage_profile()

        return graphs
