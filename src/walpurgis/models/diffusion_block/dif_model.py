"""
Walpurgis ST Localized Convolution — Graph Diffusion with Support Decay
========================================================================
Derived from D2STGNN STLocalizedConv with ~20% algorithmic changes.

Changes vs upstream:
  1. Support decay weighting: higher-order graph supports contribute with
     exponentially decreasing weight to prevent over-smoothing
  2. Residual shortcut from identity features through the gconv path
  3. Row-normalized higher-order predefined graph powers
  4. Full debug_state dict for breakpoint-style inspection
  5. Gradient hook tracking on key linear layers
"""

import torch
import torch.nn as nn
import time
from collections import deque


def _estimate_gconv_flops(n_supports, n_nodes, k_size, feat_dim, batch, seq):
    """Rough FLOPs for one gconv call: matmuls + projection."""
    mm_flops = n_supports * batch * seq * n_nodes * k_size * feat_dim * 2
    proj_flops = batch * seq * n_nodes * (n_supports + 1) * feat_dim * feat_dim * 2
    return mm_flops + proj_flops


class STLocalizedConv(nn.Module):
    """Spatial-temporal localized convolution via graph diffusion.
    
    Performs k-hop graph convolution on temporally unfolded input,
    aggregating information from predefined, dynamic, and static graphs.
    
    Walpurgis changes:
    - Support decay: w_i = exp(-0.15·i), later hops contribute less
    - Residual shortcut: output = gconv(X_k) + X_0
    - Higher-order powers are row-normalized to prevent NaN
    - self.debug_state captures full internal state for pdb inspection
    
    Debug usage:
        state = conv_layer.debug_state   # inspect after any forward()
        stats = conv_layer.timing_stats() # cumulative performance
    """
    
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None,
                 dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined = use_pre
        self.use_dynamic = dy_graph
        self.use_static = sta_graph

        self.n_graph_types = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.n_support_mats = (
            int(use_pre) * len(self.pre_defined_graph)
            + len(self.pre_defined_graph) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1

        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self._build_predefined_supports(self.pre_defined_graph)

        self.temporal_fc = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.spatial_proj = nn.Linear(self.hidden_dim * self.n_support_mats, self.hidden_dim)

        self.norm = nn.BatchNorm2d(self.hidden_dim)
        self.act = nn.ReLU()
        
        # ── Debug infrastructure ──
        self._n_calls = 0
        self._cum_ms = 0.0
        self._cum_flops = 0
        self.debug_state = {}
        self._grad_history = {'temporal_fc': deque(maxlen=100), 'spatial_proj': deque(maxlen=100)}
        
        # Register gradient hooks
        self.temporal_fc.weight.register_hook(
            lambda g: self._grad_history['temporal_fc'].append(g.norm().item())
        )
        self.spatial_proj.weight.register_hook(
            lambda g: self._grad_history['spatial_proj'].append(g.norm().item())
        )

    def _build_predefined_supports(self, graphs):
        """Build k-hop ST-localized supports from predefined adjacency matrices.
        
        Higher-order powers (k≥2) are row-normalized to prevent numerical
        explosion — upstream D2STGNN uses raw powers which blow up when the
        predefined graph has eigenvalues > 1.
        """
        all_supports = []
        diag_mask = 1 - torch.eye(graphs[0].shape[0]).to(graphs[0].device)
        
        for base_graph in graphs:
            power = base_graph
            all_supports.append(power * diag_mask)
            
            for k in range(2, self.k_s + 1):
                power = torch.matmul(base_graph, power)
                # Row-normalize to preserve scale of first-order
                row_sum = power.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                base_scale = base_graph.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                power = power / row_sum * base_scale
                all_supports.append(power * diag_mask)
        
        # Expand each graph along temporal kernel
        localized = []
        for g in all_supports:
            expanded = g.unsqueeze(-2).expand(-1, self.k_t, -1)
            flat = expanded.reshape(g.shape[0], expanded.shape[1] * expanded.shape[2])
            localized.append(flat)
        return localized

    def _graph_conv(self, supports, X_k, X_identity):
        """Graph convolution with support-decay weighting.
        
        Each support matrix contributes with weight exp(-0.15·i),
        so higher-order (more distant) neighbors have diminishing influence.
        This mitigates the over-smoothing problem in deep GCN stacks.
        
        Also adds a residual shortcut from X_identity (mean over temporal
        kernel) for gradient health through deep decouple layers.
        """
        aggregated = [X_identity]  # identity feature as first element
        norms = []
        _decay = 0.15
        
        for idx, graph in enumerate(supports):
            if graph.dim() == 2:
                pass  # [N, k_t*N] — predefined
            else:
                graph = graph.unsqueeze(1)  # add seq dim
            
            neighbor_agg = torch.matmul(graph, X_k)
            
            # Apply exponential decay weight
            decay_w = torch.exp(torch.tensor(-_decay * idx,
                                             device=neighbor_agg.device,
                                             dtype=neighbor_agg.dtype))
            neighbor_agg = neighbor_agg * decay_w
            
            norms.append(neighbor_agg.norm().item())
            aggregated.append(neighbor_agg)
        
        self.debug_state['support_norms'] = norms
        self.debug_state['decay_weights'] = [
            float(torch.exp(torch.tensor(-_decay * i)).item())
            for i in range(len(supports))
        ]
        
        concat = torch.cat(aggregated, dim=-1)
        self.debug_state['concat_shape'] = list(concat.shape)
        
        out = self.spatial_proj(concat)
        out = self.dropout(out)
        
        # Residual shortcut from identity features
        out = out + X_identity
        return out

    def timing_stats(self):
        """Return cumulative timing and FLOP statistics — call from debugger."""
        avg = self._cum_ms / max(self._n_calls, 1)
        return {
            'calls': self._n_calls,
            'total_ms': self._cum_ms,
            'avg_ms': avg,
            'total_flops': self._cum_flops,
            'recent_grads': {
                k: list(v)[-5:] for k, v in self._grad_history.items()
            }
        }

    def forward(self, X, dynamic_graph, static_graph):
        """Forward with full debug instrumentation.
        
        After calling, inspect self.debug_state for shapes, norms, timing.
        """
        self._n_calls += 1
        t0 = time.perf_counter()
        verbose = (self._n_calls % 500 == 1)
        
        self.debug_state['input_shape'] = list(X.shape)
        self.debug_state['input_norm'] = X.norm().item()
        
        # Temporal unfolding
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, S, N, K, F = X.shape
        self.debug_state['unfolded'] = [B, S, N, K, F]
        
        if verbose:
            print(f"        [STConv #{self._n_calls}] unfolded: [{B},{S},{N},{K},{F}]")
        
        # Collect support set
        supports = []
        if self.use_predefined:
            supports.extend(self.pre_defined_graph)
        if self.use_dynamic:
            supports.extend(dynamic_graph)
        if self.use_static:
            supports.extend(self._build_predefined_supports(static_graph))
        
        self.debug_state['n_supports'] = len(supports)
        
        # Temporal FC
        X_flat = X.reshape(B, S, N, K * F)
        temporal_out = self.temporal_fc(X_flat)
        temporal_out = self.act(temporal_out)
        self.debug_state['temporal_fc_norm'] = temporal_out.norm().item()
        
        # Prepare identity and kernel features
        temporal_out = temporal_out.view(B, S, N, K, F)
        X_identity = torch.mean(temporal_out, dim=-2)
        X_kernel = temporal_out.transpose(-3, -2).reshape(B, S, K * N, F)
        
        # Graph convolution
        hidden = self._graph_conv(supports, X_kernel, X_identity)
        
        elapsed = (time.perf_counter() - t0) * 1000
        self._cum_ms += elapsed
        
        flops = _estimate_gconv_flops(len(supports), N, K, F, B, S)
        self._cum_flops += flops
        
        self.debug_state['output_shape'] = list(hidden.shape)
        self.debug_state['output_norm'] = hidden.norm().item()
        self.debug_state['elapsed_ms'] = elapsed
        
        if verbose:
            avg = self._cum_ms / self._n_calls
            print(f"        [STConv] out={list(hidden.shape)} "
                  f"supports={len(supports)} {elapsed:.2f}ms (avg={avg:.2f}ms) "
                  f"FLOPs≈{flops/1e6:.1f}M")
        
        return hidden
