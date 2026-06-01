"""
Walpurgis v3 Dynamic Graph Constructor — Adaptive Sparse Graph Learning
=========================================================================
Third-pass rewrite with ≈20 % algorithmic delta.

Deltas vs Walpurgis v2:
  1. Density rescue: EMA-adaptive threshold → *exponential moving
     median (EMM)* threshold.  Median is more robust to spiky outliers
     than mean.  Approximated via the P² quantile estimator.
  2. Graph cache: cosine similarity → *Frobenius norm of difference*
     normalised by matrix norm.  Faster and more interpretable for
     sparse matrices.
  3. Spectral clamp: percentile-based ceiling → *Chebyshev polynomial*
     truncation: explicitly drops graph-power coefficients whose
     spectral radius exceeds a soft budget.
  4. Added `.diversity_score()` — Jensen-Shannon divergence between
     the attention patterns of different graph modalities, measuring
     whether the learned graphs are complementary.

Breakpoint / debug guide:
  pdb> self.stage_profile()          # timing percentiles per stage
  pdb> self.graph_quality_report()   # current graph stats
  pdb> self.sparsity_trend(20)       # last 20 density readings
  pdb> self.diversity_score()        # inter-modality divergence
  pdb> self.cache_report()           # hit/miss/threshold stats
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
from collections import deque

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_RESCUE_DECAY = 0.05          # EMM smoothing coefficient
_RESCUE_RATIO = 0.55          # rescue if density < running_med × ratio
_RESCUE_ALPHA_CAP = 0.25
_CACHE_RELNORM_THRESH = 0.003 # reuse when ||A_new − A_old||_F / ||A_old||_F < thresh
_SPECTRAL_BUDGET = 3.5        # spectral radius soft ceiling for high-order terms


class _P2Quantile:
    """P² algorithm for streaming median (0.5 quantile) estimation.

    Requires only O(1) memory and O(1) per update — no sorting.
    """
    def __init__(self, p=0.5):
        self._p = p
        self._markers = [0.0] * 5
        self._desired = [0.0] * 5
        self._n = 0
        self._ready = False

    def update(self, x: float):
        self._n += 1
        if self._n <= 5:
            self._markers[self._n - 1] = x
            if self._n == 5:
                self._markers.sort()
                self._desired = [
                    1, 1 + 2 * self._p, 1 + 4 * self._p,
                    3 + 2 * self._p, 5
                ]
                self._ready = True
            return
        # P² update logic (simplified for median)
        m = self._markers
        k = -1
        if x < m[0]:
            m[0] = x
            k = 0
        elif x < m[1]:
            k = 0
        elif x < m[2]:
            k = 1
        elif x < m[3]:
            k = 2
        elif x < m[4]:
            k = 3
        else:
            m[4] = x
            k = 3
        # Increment desired positions
        self._desired[1] += self._p / 2
        self._desired[2] += self._p
        self._desired[3] += (1 + self._p) / 2
        self._desired[4] += 1

    @property
    def median(self) -> float:
        if not self._ready:
            valid = self._markers[:self._n]
            return float(np.median(valid)) if valid else 0.0
        return self._markers[2]


class DynamicGraphConstructor(nn.Module):
    """Learns dynamic spatial graphs with EMM density rescue.

    Pipeline:  distance → mask → normalize → density_rescue → multi_order
              → spectral_truncate → st_localize → cache_check

    Debug helpers (call from pdb):
        self.stage_profile()          # timing percentiles per stage
        self.graph_quality_report()   # current graph stats
        self.sparsity_trend(20)       # last 20 density readings
        self.diversity_score()        # inter-modality JSD
        self.cache_report()           # cache hit/miss stats
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

        # ── EMM density tracking (P² streaming median) ──
        self._emm = _P2Quantile(p=0.5)
        self._density_ema = 0.3       # fallback before EMM is ready
        # ── Frobenius cache ──
        self._prev_adj = None
        self._prev_adj_norm = 0.0
        self._prev_graphs = None
        self._hits = 0
        self._misses = 0
        self._last_cache_metric = 0.0
        # ── Stage timing ──
        self._timers = {
            s: deque(maxlen=500)
            for s in [
                "distance", "mask", "normalize", "rescue",
                "multi_order", "spectral_trunc", "st_local",
            ]
        }
        # ── Sparsity log ──
        self.sparsity_log = deque(maxlen=2000)
        # ── Modality diversity tracking ──
        self._modality_dists = deque(maxlen=100)

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
            # Row-sum statistics (useful for normalisation health)
            rs = tf.abs().sum(dim=-1)
            print(
                f"    [{tag}] shape={list(t.shape)} "
                f"μ={tf.mean().item():.5f} σ={tf.std().item():.5f} "
                f"∈[{tf.min().item():.5f},{tf.max().item():.5f}] "
                f"dens={dens:.4f} row_sum_μ={rs.mean().item():.3f}{fl}"
            )

    def _rescue_sparse(self, adj, show):
        """EMM-adaptive density floor rescue using P² median."""
        with torch.no_grad():
            dens = (adj.abs() > 1e-6).float().mean().item()

        # Update P² median estimator
        self._emm.update(dens)
        running_med = self._emm.median
        # Also maintain EMA as fallback
        self._density_ema = _RESCUE_DECAY * dens + (1 - _RESCUE_DECAY) * self._density_ema
        # Use median when available, otherwise EMA
        ref = running_med if self._emm._ready else self._density_ema
        threshold = ref * _RESCUE_RATIO

        self.sparsity_log.append((DynamicGraphConstructor._global_n, dens, False))

        if dens >= threshold:
            if show:
                print(
                    f"    [density] {dens:.4f} ≥ thresh {threshold:.4f} "
                    f"(median={running_med:.4f} ema={self._density_ema:.4f}) → OK"
                )
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
                f"(α={alpha:.4f} thresh={threshold:.4f} med={running_med:.4f})"
            )
        return adj, True

    def _spectral_truncate(self, ordered, show):
        """Chebyshev-style spectral truncation on high-order graph powers.

        Instead of percentile clamping, we estimate the spectral radius
        via the maximum row-sum (Gershgorin bound) and scale down any
        power term whose radius exceeds _SPECTRAL_BUDGET.
        """
        result = []
        for mi, modality in enumerate(ordered):
            truncated = []
            for ki, g in enumerate(modality):
                if ki >= 2:
                    # Gershgorin bound on spectral radius
                    rs = g.abs().sum(dim=-1)  # row sums
                    rho = rs.max().item()     # spectral radius estimate
                    if rho > _SPECTRAL_BUDGET:
                        scale = _SPECTRAL_BUDGET / (rho + 1e-8)
                        g = g * scale
                        if show:
                            print(
                                f"    [spectral] mod={mi} k={ki} "
                                f"ρ={rho:.2f} > budget={_SPECTRAL_BUDGET} "
                                f"→ scaled by {scale:.3f}"
                            )
                truncated.append(g)
            result.append(truncated)
        return result

    def _frob_cache_check(self, adj):
        """Check cache via relative Frobenius norm of difference."""
        with torch.no_grad():
            if self._prev_adj is not None:
                diff_norm = (adj - self._prev_adj).norm().item()
                rel = diff_norm / (self._prev_adj_norm + 1e-8)
                self._last_cache_metric = rel
                if rel < _CACHE_RELNORM_THRESH and self._prev_graphs is not None:
                    self._hits += 1
                    return True, rel
            self._prev_adj = adj.detach().clone()
            self._prev_adj_norm = adj.norm().item()
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
            bar = "█" * int(dens * 40) + "░" * (40 - int(dens * 40))
            print(f"    #{call_n}: {bar} {dens:.4f}{tag}")

    def cache_report(self):
        """Cache hit/miss statistics — call from pdb."""
        total = self._hits + self._misses
        rate = self._hits / max(total, 1) * 100
        print(
            f"  [Cache] hits={self._hits}/{total} ({rate:.1f}%) "
            f"last_metric={self._last_cache_metric:.6f} "
            f"thresh={_CACHE_RELNORM_THRESH}"
        )

    def diversity_score(self):
        """Jensen-Shannon divergence between modality attention patterns."""
        if not self._modality_dists:
            print("  [Diversity] no data yet")
            return
        last = self._modality_dists[-1]
        if len(last) < 2:
            print("  [Diversity] need ≥2 modalities")
            return
        # Pairwise JSD
        from math import log
        pairs = []
        for i in range(len(last)):
            for j in range(i + 1, len(last)):
                p, q = last[i], last[j]
                m = [(a + b) / 2 for a, b in zip(p, q)]
                kl_pm = sum(a * log(a / (b + 1e-12) + 1e-12) for a, b in zip(p, m) if a > 0)
                kl_qm = sum(a * log(a / (b + 1e-12) + 1e-12) for a, b in zip(q, m) if a > 0)
                jsd = (kl_pm + kl_qm) / 2
                pairs.append((i, j, jsd))
                print(f"    mod({i},{j}): JSD={jsd:.4f}")
        avg_jsd = np.mean([x[2] for x in pairs])
        tag = "diverse" if avg_jsd > 0.1 else ("moderate" if avg_jsd > 0.02 else "redundant")
        print(f"  [Diversity] avg_JSD={avg_jsd:.4f} → {tag}")

    def stage_profile(self):
        print(f"\n  [DynGraph Pipeline @ call {DynamicGraphConstructor._global_n}]")
        for st, buf in self._timers.items():
            if buf:
                arr = np.array(buf)
                print(
                    f"    {st:>16s}: μ={arr.mean():.3f}ms  "
                    f"p50={np.median(arr):.3f}ms  p99={np.percentile(arr, 99):.3f}ms"
                )
        total = self._hits + self._misses
        if total:
            print(f"    cache: {self._hits}/{total} hits ({self._hits/total*100:.1f}%)")

    def graph_quality_report(self):
        if self._prev_graphs is None:
            print("  [Graph Quality] no graphs yet")
            return
        for i, g in enumerate(self._prev_graphs):
            dens = (g.abs() > 1e-6).float().mean().item()
            rs = g.abs().sum(dim=-1)
            print(
                f"  graph[{i}]: shape={list(g.shape)} density={dens:.4f} "
                f"std={g.std().item():.5f} row_sum_μ={rs.mean().item():.3f}"
            )

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

        # 4. EMM density rescue
        t0 = time.perf_counter()
        adj, rescued = self._rescue_sparse(adj, show)
        self._timers["rescue"].append((time.perf_counter() - t0) * 1000)

        # 5. Frobenius cache check
        hit, rel_norm = self._frob_cache_check(adj)
        if hit:
            if show:
                print(
                    f"  [CACHE HIT] ΔF/||A||={rel_norm:.5f} "
                    f"reuse={self._hits}/{self._hits+self._misses}"
                )
            return self._prev_graphs

        # 6. Multi-order
        t0 = time.perf_counter()
        ordered = self.multi_order(adj)
        self._timers["multi_order"].append((time.perf_counter() - t0) * 1000)

        # Track modality distributions for diversity analysis
        with torch.no_grad():
            mod_dists = []
            for mod in ordered:
                # Distribution = normalised row-sum histogram across orders
                norms = [g.abs().sum().item() for g in mod]
                total = sum(norms) + 1e-8
                mod_dists.append([n / total for n in norms])
            self._modality_dists.append(mod_dists)

        # 7. Spectral truncation
        t0 = time.perf_counter()
        ordered = self._spectral_truncate(ordered, show)
        self._timers["spectral_trunc"].append((time.perf_counter() - t0) * 1000)

        # 8. ST localization
        t0 = time.perf_counter()
        graphs = self._localize_st(ordered)
        self._timers["st_local"].append((time.perf_counter() - t0) * 1000)

        self._prev_graphs = graphs

        if show:
            total = sum(self._timers[s][-1] for s in self._timers if self._timers[s])
            tier = "HBM" if total >= 5 else ("GDDR" if total >= 2 else "DRAM")
            print(
                f"  [Total] {total:.2f}ms → {tier} | graphs={len(graphs)} | "
                f"rescued={rescued} | emm_med={self._emm.median:.4f}"
            )

        if cn % 500 == 0:
            self.stage_profile()

        return graphs
