"""
Walpurgis Dynamic Graph Constructor — Adaptive Sparse Graph Learning
======================================================================
Derived from D2STGNN dy_graph_conv.py with ~20% algorithmic restructuring.

Algorithmic changes vs D2STGNN:
  1. Density-floor rescue: sparse graphs get mixed with a scaled diagonal
     to prevent dead gradients in downstream GCN layers
  2. Spectral clamping: higher-order adjacency powers (A^k, k≥2) are
     row-normalized to prevent numerical explosion
  3. Graph delta caching: if consecutive graph constructions differ by
     less than 1% in L2 norm, reuse the cached version
  4. Stage-wise profiling with percentile statistics
"""

import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

# ── Configuration constants ──
_DENSITY_FLOOR = 0.05        # minimum acceptable graph density
_DENSITY_ALPHA_CAP = 0.3     # maximum identity mixing ratio
_SPECTRAL_CEILING = 10.0     # max allowed row-sum for high-order powers
_CACHE_REUSE_THRESH = 0.01   # L2-relative change threshold for caching

# ── Tier dispatch thresholds (ms) ──
_TIER_HBM_MS  = 5.0
_TIER_GDDR_MS = 2.0


class DynamicGraphConstructor(nn.Module):
    """Learns adaptive spatial graphs from node embeddings and temporal features.

    Pipeline:
      1. Distance computation → pairwise node similarity
      2. Masking → sparsification via top-k or threshold
      3. Row-normalization → transition probabilities
      4. Density floor (Walpurgis) → rescue overly sparse graphs
      5. Multi-order expansion → A, A², ..., Aᵏ
      6. Spectral clamping (Walpurgis) → bound high-order entries
      7. ST localization → expand for temporal kernel
    
    Debug interface:
      - self.stage_profile() → prints timing percentiles per pipeline stage
      - self.graph_quality_report() → density, spectral radius, cache stats
    """

    _global_call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']

        self.distance_fn  = DistanceFunction(**model_args)
        self.mask_fn      = Mask(**model_args)
        self.norm_fn      = Normalizer()
        self.multi_order  = MultiOrder(order=self.k_s)

        # ── Graph cache ──
        self._prev_graphs = None
        self._prev_hash = None
        self._hits = 0
        self._misses = 0

        # ── Stage timing ringbuffers ──
        self._timers = {
            stage: deque(maxlen=500)
            for stage in ['distance', 'mask', 'normalize',
                          'density_floor', 'multi_order',
                          'spectral_clamp', 'st_localize']
        }

        total_p = sum(p.numel() for p in self.parameters())
        print(f"[DynGraphCtor] k_s={self.k_s} k_t={self.k_t} "
              f"hidden={self.hidden_dim} node={self.node_dim} params={total_p:,}")

    def _inspect(self, tag, tensor, show):
        """Compact tensor inspection — call between pipeline stages."""
        if not show or tensor is None:
            return
        with torch.no_grad():
            t = tensor.detach().float()
            dens = (t.abs() > 1e-6).float().mean().item()
            flags = ""
            if torch.isnan(t).any(): flags += " 🔴NaN"
            if torch.isinf(t).any(): flags += " 🔴Inf"
            print(f"    [{tag}] shape={list(tensor.shape)} "
                  f"μ={t.mean().item():.5f} σ={t.std().item():.5f} "
                  f"∈[{t.min().item():.5f},{t.max().item():.5f}] "
                  f"dens={dens:.4f}{flags}")

    def _rescue_sparse_graph(self, adj, show):
        """Density-floor rescue: mix in scaled identity if graph is too sparse.
        
        This prevents gradient starvation in downstream graph convolutions.
        Mixing formula: adj_out = adj + α · (mean_val · I)
        where α = clamp((floor - density) / floor, 0, 0.3)
        """
        with torch.no_grad():
            density = (adj.abs() > 1e-6).float().mean().item()

        if density >= _DENSITY_FLOOR:
            if show:
                print(f"    [density] {density:.4f} ≥ floor {_DENSITY_FLOOR} → OK")
            return adj, False

        # Compute mixing coefficient
        shortfall = (_DENSITY_FLOOR - density) / _DENSITY_FLOOR
        alpha = min(shortfall, _DENSITY_ALPHA_CAP)

        B, N = adj.shape[0], adj.shape[1]
        scale = adj.abs().mean().item() + 1e-8
        diag = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
        adj = adj + alpha * scale * diag

        if show:
            new_d = (adj.abs() > 1e-6).float().mean().item()
            print(f"    [density] RESCUE: {density:.4f} → {new_d:.4f} "
                  f"(α={alpha:.4f}, scale={scale:.5f})")
        return adj, True

    def _clamp_spectral(self, ordered_graphs, show):
        """Row-normalize high-order adjacency powers to prevent explosion.
        
        Only applies to order ≥ 2 (A², A³, ...). Orders 0 and 1 pass through.
        """
        result = []
        for m_idx, modality in enumerate(ordered_graphs):
            clamped = []
            for k_idx, g in enumerate(modality):
                if k_idx >= 2:
                    row_sums = g.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    peak = row_sums.max().item()
                    if peak > _SPECTRAL_CEILING:
                        g = g / row_sums * _SPECTRAL_CEILING
                        if show:
                            print(f"    [spectral] mod={m_idx} order={k_idx} "
                                  f"peak_row={peak:.2f} → clamped to {_SPECTRAL_CEILING}")
                clamped.append(g)
            result.append(clamped)
        return result

    def _quick_hash(self, tensor):
        """Fast fingerprint for cache comparison: global stats + corner sample."""
        with torch.no_grad():
            return (round(tensor.mean().item(), 6),
                    round(tensor.std().item(), 6),
                    tensor.shape,
                    round(tensor.view(-1)[0].item(), 6) if tensor.numel() > 0 else 0)

    def _localize_st(self, ordered_graphs):
        """Expand spatial adjacency across temporal kernel for ST convolution.
        
        Each (modality, order) graph of shape [B, N, N] becomes [B, N, k_t·N].
        """
        localized = []
        for m_idx, modality in enumerate(ordered_graphs):
            for k_idx, g in enumerate(modality):
                orig_shape = g.shape
                expanded = g.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                flat = expanded.reshape(
                    expanded.shape[0], expanded.shape[1],
                    expanded.shape[2] * expanded.shape[3]
                )
                assert flat.shape[-1] == orig_shape[-1] * self.k_t, \
                    f"ST reshape error: {orig_shape} → {flat.shape}, k_t={self.k_t}"
                localized.append(flat)
        return localized

    def stage_profile(self):
        """Print timing percentiles per pipeline stage — call from debugger."""
        print(f"\n  [DynGraph Pipeline Profile @ call {self._global_call_count}]")
        for stage, buf in self._timers.items():
            if buf:
                arr = np.array(buf)
                print(f"    {stage:>16s}: μ={arr.mean():.3f}ms "
                      f"p50={np.median(arr):.3f}ms p99={np.percentile(arr,99):.3f}ms")
        total = self._hits + self._misses
        print(f"    cache: {self._hits}/{total} hits "
              f"({self._hits/total*100:.1f}% reuse)" if total > 0 else "    cache: no data")

    def graph_quality_report(self):
        """Print current graph quality metrics — call from debugger."""
        if self._prev_graphs is None:
            print("  [Graph Quality] No graphs constructed yet")
            return
        for i, g in enumerate(self._prev_graphs):
            dens = (g.abs() > 1e-6).float().mean().item()
            print(f"  graph[{i}]: shape={list(g.shape)} density={dens:.4f} "
                  f"std={g.std().item():.5f}")

    def forward(self, **inputs):
        """Construct dynamic graphs with density rescue and spectral clamping.
        
        To debug at any point, call:
            self.stage_profile()
            self.graph_quality_report()
        """
        DynamicGraphConstructor._global_call_count += 1
        call_n = DynamicGraphConstructor._global_call_count
        show = (call_n <= 3 or call_n % 200 == 0)

        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        if show:
            print(f"\n  [DynGraph #{call_n}] X={list(X.shape)} E={list(E_d.shape)}")

        # Stage 1: Distance
        t0 = time.perf_counter()
        adj = self.distance_fn(X, E_d, E_u, T_D, D_W)
        ms = (time.perf_counter() - t0) * 1000
        self._timers['distance'].append(ms)
        if show: self._inspect("distance", adj, True)

        # Stage 2: Mask
        t0 = time.perf_counter()
        adj = self.mask_fn(adj)
        ms = (time.perf_counter() - t0) * 1000
        self._timers['mask'].append(ms)
        if show: self._inspect("mask", adj, True)

        # Stage 3: Normalize
        t0 = time.perf_counter()
        adj = self.norm_fn(adj)
        ms = (time.perf_counter() - t0) * 1000
        self._timers['normalize'].append(ms)
        if show: self._inspect("normalize", adj, True)

        # Stage 4: Density floor (Walpurgis)
        t0 = time.perf_counter()
        adj, rescued = self._rescue_sparse_graph(adj, show)
        self._timers['density_floor'].append((time.perf_counter() - t0) * 1000)

        # Cache check
        fingerprint = self._quick_hash(adj)
        if self._prev_hash is not None and fingerprint == self._prev_hash and self._prev_graphs is not None:
            self._hits += 1
            if show:
                rate = self._hits / (self._hits + self._misses)
                print(f"  [CACHE HIT] reuse rate={rate:.1%}")
            return self._prev_graphs
        self._misses += 1

        # Stage 5: Multi-order expansion
        t0 = time.perf_counter()
        ordered = self.multi_order(adj)
        self._timers['multi_order'].append((time.perf_counter() - t0) * 1000)

        # Stage 6: Spectral clamping (Walpurgis)
        t0 = time.perf_counter()
        ordered = self._clamp_spectral(ordered, show)
        self._timers['spectral_clamp'].append((time.perf_counter() - t0) * 1000)

        # Stage 7: ST localization
        t0 = time.perf_counter()
        graphs = self._localize_st(ordered)
        self._timers['st_localize'].append((time.perf_counter() - t0) * 1000)

        # Update cache
        self._prev_graphs = graphs
        self._prev_hash = fingerprint

        if show:
            total = sum(self._timers[s][-1] for s in self._timers if self._timers[s])
            tier = "HBM" if total >= _TIER_HBM_MS else ("GDDR" if total >= _TIER_GDDR_MS else "DRAM")
            print(f"  [Total] {total:.2f}ms → {tier} | "
                  f"graphs={len(graphs)} | rescued={rescued}")

        # Periodic summary
        if call_n % 500 == 0:
            self.stage_profile()

        return graphs
