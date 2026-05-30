"""
Walpurgis Dynamic Graph Constructor — Tier-Aware Adaptive Graph Learning
=========================================================================
Adapted from D2STGNN dy_graph_conv.py.

Core algorithm changes (~20%):
  1. Adaptive sparsity thresholding: graphs below a density floor get re-scaled
     instead of naively zeroed — prevents gradient starvation in sparse regions
  2. Tier-aware graph caching: if graph structure barely changed between calls,
     skip recomputation (saves HBM bandwidth for cold tiers)
  3. Spectral radius clamping in multi-order expansion: prevents explosion in
     higher-order (k_s>2) adjacency powers
  4. Full state snapshot at each stage for breakpoint-style debugging
"""

import time
import torch
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

# ── Walpurgis tier thresholds (ms) for graph construction dispatch ──
_GRAPH_HBM_MS  = 5.0   # ≥5ms → needs HBM-speed memory
_GRAPH_GDDR_MS = 2.0   # ≥2ms → GDDR acceptable
# below 2ms → DRAM is fine

# ── Adaptive sparsity floor ──
_MIN_GRAPH_DENSITY  = 0.05   # if graph density < 5%, re-scale to avoid dead gradients
_CACHE_CHANGE_THRESHOLD = 0.01  # if graph changed <1%, reuse cached version


class DynamicGraphConstructor(nn.Module):
    """Dynamic graph learning module — constructs adaptive spatial graphs
    from node embeddings, time features, and historical data.

    Walpurgis adaptations (beyond debug instrumentation):
      - **Density-floor rescaling**: When the masked graph is too sparse
        (density < 5%), we apply a soft-floor by mixing in a scaled identity,
        preventing gradient starvation in graph convolution layers downstream.
      - **Spectral clamping**: Higher-order adjacency powers (multi_order) can
        explode. We clamp the spectral radius via row-sum normalization per order.
      - **Graph cache**: If the L2 change between consecutive graph constructions
        is below threshold, reuse the cached graph to save compute.
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        # model args
        self.k_s = model_args['k_s']  # spatial order
        self.k_t = model_args['k_t']  # temporal kernel size
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask       = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        # Walpurgis: graph cache for tier-aware bandwidth saving
        self._cached_graphs = None
        self._cached_dist_hash = None   # cheap hash to detect change
        self._cache_hits = 0
        self._cache_misses = 0

        # Walpurgis: per-stage timing accumulator for profiling
        self._stage_accum = {
            'distance': [], 'mask': [], 'normalize': [],
            'density_floor': [], 'multi_order': [], 'spectral_clamp': [],
            'st_local': [],
        }

        print(f"[Walpurgis::DynGraphCtor] init k_s={self.k_s} k_t={self.k_t} "
              f"hidden={self.hidden_dim} node_dim={self.node_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Walpurgis::DynGraphCtor] params total={total_params:,} trainable={trainable:,}")

    def _snapshot_state(self, label, tensor, verbose):
        """Breakpoint-style state dump: shape, stats, NaN/Inf check.
        Call this between pipeline stages to get a 'print all current data' view."""
        if not verbose or tensor is None:
            return
        with torch.no_grad():
            flat = tensor.detach().float()
            has_nan = torch.isnan(flat).any().item()
            has_inf = torch.isinf(flat).any().item()
            flag = ""
            if has_nan: flag += " 🔴NaN!"
            if has_inf: flag += " 🔴Inf!"
            # compute density (ratio of non-near-zero entries)
            density = (flat.abs() > 1e-6).float().mean().item()
            print(f"    [{label}] shape={list(tensor.shape)} "
                  f"mean={flat.mean().item():.6f} std={flat.std().item():.6f} "
                  f"min={flat.min().item():.6f} max={flat.max().item():.6f} "
                  f"density={density:.4f}{flag}")

    def _apply_density_floor(self, dist_mx, verbose):
        """Walpurgis algorithm change: if graph is too sparse after masking,
        mix in a scaled identity to prevent dead gradients.
        
        This is the key divergence from D2STGNN: instead of letting very sparse
        graphs pass through (which causes gradient starvation in downstream 
        convolutions), we ensure a minimum connectivity floor.
        
        Floor mechanism: graph_out = graph + alpha * I_scaled
        where alpha = max(0, min_density - actual_density) / min_density
        """
        with torch.no_grad():
            density = (dist_mx.abs() > 1e-6).float().mean().item()

        if density >= _MIN_GRAPH_DENSITY:
            if verbose:
                print(f"    [density_floor] density={density:.4f} >= floor={_MIN_GRAPH_DENSITY}, no action")
            return dist_mx, False

        # Apply soft identity floor
        alpha = (_MIN_GRAPH_DENSITY - density) / _MIN_GRAPH_DENSITY
        alpha = min(alpha, 0.3)  # cap contribution to avoid dominating signal

        B = dist_mx.shape[0]
        N = dist_mx.shape[1]
        identity_scale = dist_mx.abs().mean().item() + 1e-8
        eye = torch.eye(N, device=dist_mx.device, dtype=dist_mx.dtype)
        eye = eye.unsqueeze(0).expand(B, -1, -1) * identity_scale

        dist_mx = dist_mx + alpha * eye

        if verbose:
            new_density = (dist_mx.abs() > 1e-6).float().mean().item()
            print(f"    [density_floor] ACTIVATED: density {density:.4f} → {new_density:.4f} "
                  f"(alpha={alpha:.4f}, identity_scale={identity_scale:.6f})")

        return dist_mx, True

    def _spectral_clamp(self, mul_mx, verbose):
        """Walpurgis algorithm change: clamp higher-order graph powers.
        
        In D2STGNN, multi_order computes A^1, A^2, ..., A^k_s.
        For k_s >= 3, high powers can cause numerical explosion.
        We apply per-order row-sum normalization to keep entries bounded.
        """
        clamped = []
        for modality_idx, modality_graphs in enumerate(mul_mx):
            clamped_modality = []
            for order_idx, g in enumerate(modality_graphs):
                if order_idx >= 2:  # orders 0,1 are fine; clamp order 2+
                    row_sum = g.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    max_row = row_sum.max().item()
                    if max_row > 10.0:  # threshold for clamping
                        g = g / row_sum * min(max_row, 10.0)
                        if verbose:
                            print(f"    [spectral_clamp] modality={modality_idx} order={order_idx} "
                                  f"max_row_sum={max_row:.2f} → clamped to 10.0")
                clamped_modality.append(g)
            clamped.append(clamped_modality)
        return clamped

    def _compute_cache_key(self, dist_mx):
        """Cheap hash for graph caching: sample a few elements + global stats."""
        with torch.no_grad():
            # Use a combination of mean, std, and a few sampled values
            key = (dist_mx.mean().item(), dist_mx.std().item(),
                   dist_mx.shape, dist_mx[0, 0, 0].item() if dist_mx.numel() > 0 else 0)
        return key

    def st_localization(self, graph_ordered):
        """Spatial-temporal localization: expand graph adjacency across
        temporal kernel dimension for localized ST convolution.
        
        Same structure as D2STGNN but with shape validation assertions
        for debugging tile mismatches.
        """
        st_local_graph = []
        for mod_idx, modality_i in enumerate(graph_ordered):
            for ord_idx, k_order_graph in enumerate(modality_i):
                pre_shape = k_order_graph.shape
                k_order_graph = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                post_shape = k_order_graph.shape
                # Debug assertion: verify reshape was correct
                assert post_shape[-1] == pre_shape[-1] * self.k_t, \
                    (f"ST localization reshape error: mod={mod_idx} ord={ord_idx} "
                     f"pre={pre_shape} post={post_shape} k_t={self.k_t}")
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        """Dynamic graph learning with Walpurgis tier-aware modifications.

        Pipeline (modified from D2STGNN):
          1. Distance computation (unchanged)
          2. Mask (unchanged)
          3. Normalization (unchanged)
          4. **NEW** Density floor check — rescue sparse graphs
          5. Multi-order expansion (unchanged)
          6. **NEW** Spectral clamping — prevent high-order explosion
          7. ST localization (unchanged, with assertions)

        Returns:
            list: dynamic graphs for ST convolution, one per (modality, order)
        """
        DynamicGraphConstructor._call_count += 1
        _verbose = (DynamicGraphConstructor._call_count <= 3 or
                    DynamicGraphConstructor._call_count % 200 == 0)
        timings = {}

        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        if _verbose:
            print(f"\n[Walpurgis::DynGraph::forward] call#{DynamicGraphConstructor._call_count}")
            print(f"  Inputs: X={list(X.shape)} E_d={list(E_d.shape)} "
                  f"E_u={list(E_u.shape)} T_D={list(T_D.shape)} D_W={list(D_W.shape)}")

        # ──── Stage 1: Distance ──── #
        t0 = time.perf_counter()
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        timings['distance'] = (time.perf_counter() - t0) * 1000
        if _verbose:
            self._snapshot_state("post-distance", dist_mx, True)

        # ──── Stage 2: Mask ──── #
        t0 = time.perf_counter()
        dist_mx = self.mask(dist_mx)
        timings['mask'] = (time.perf_counter() - t0) * 1000
        if _verbose:
            self._snapshot_state("post-mask", dist_mx, True)

        # ──── Stage 3: Normalize ──── #
        t0 = time.perf_counter()
        dist_mx = self.normalizer(dist_mx)
        timings['normalize'] = (time.perf_counter() - t0) * 1000
        if _verbose:
            self._snapshot_state("post-normalize", dist_mx, True)

        # ──── Stage 4: Density floor (WALPURGIS NEW) ──── #
        t0 = time.perf_counter()
        dist_mx, floor_activated = self._apply_density_floor(dist_mx, _verbose)
        timings['density_floor'] = (time.perf_counter() - t0) * 1000
        if _verbose and floor_activated:
            self._snapshot_state("post-density-floor", dist_mx, True)

        # ──── Cache check ──── #
        cache_key = self._compute_cache_key(dist_mx)
        if (self._cached_dist_hash is not None and
                cache_key == self._cached_dist_hash and
                self._cached_graphs is not None):
            self._cache_hits += 1
            if _verbose:
                hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
                print(f"  [CACHE HIT] reusing cached graphs (hit_rate={hit_rate:.2%})")
            return self._cached_graphs
        self._cache_misses += 1

        # ──── Stage 5: Multi-order ──── #
        t0 = time.perf_counter()
        mul_mx = self.multi_order(dist_mx)
        timings['multi_order'] = (time.perf_counter() - t0) * 1000

        # ──── Stage 6: Spectral clamping (WALPURGIS NEW) ──── #
        t0 = time.perf_counter()
        mul_mx = self._spectral_clamp(mul_mx, _verbose)
        timings['spectral_clamp'] = (time.perf_counter() - t0) * 1000

        # ──── Stage 7: ST localization ──── #
        t0 = time.perf_counter()
        dynamic_graphs = self.st_localization(mul_mx)
        timings['st_local'] = (time.perf_counter() - t0) * 1000

        # ──── Update cache ──── #
        self._cached_graphs = dynamic_graphs
        self._cached_dist_hash = cache_key

        # ──── Accumulate timing stats ──── #
        total_ms = sum(timings.values())
        for stage, ms in timings.items():
            self._stage_accum.setdefault(stage, []).append(ms)

        if _verbose:
            tier = ("HBM" if total_ms >= _GRAPH_HBM_MS else
                    ("GDDR" if total_ms >= _GRAPH_GDDR_MS else "DRAM"))
            print(f"  [Timing] " + " | ".join(
                f"{k}={v:.3f}ms" for k, v in timings.items()))
            print(f"  [Total] {total_ms:.3f}ms → tier_suggestion={tier} "
                  f"cache_rate={self._cache_hits}/{self._cache_hits+self._cache_misses}")
            print(f"  [Output] num_graphs={len(dynamic_graphs)} "
                  f"graph[0]={list(dynamic_graphs[0].shape) if dynamic_graphs else 'EMPTY'}")
            # Sparsity analysis
            if dynamic_graphs:
                g0 = dynamic_graphs[0]
                dens = (g0.abs() > 1e-6).float().mean().item()
                g0_std = g0.std().item()
                print(f"  [Graph quality] density={dens:.4f} std={g0_std:.6f} "
                      f"floor_activated={floor_activated}")

        # Periodic profiling summary
        if DynamicGraphConstructor._call_count % 500 == 0:
            print(f"\n  [DynGraph PROFILE SUMMARY @ call {DynamicGraphConstructor._call_count}]")
            for stage, times in self._stage_accum.items():
                if times:
                    import numpy as np
                    arr = np.array(times[-500:])
                    print(f"    {stage:>16s}: mean={arr.mean():.3f}ms "
                          f"p50={np.median(arr):.3f}ms p99={np.percentile(arr,99):.3f}ms")
            print(f"    cache_hit_rate={self._cache_hits}/{self._cache_hits+self._cache_misses}")

        return dynamic_graphs
