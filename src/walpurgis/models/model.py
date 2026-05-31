"""
Walpurgis-TSH: Decoupled Spatial-Temporal Heterogeneous-Memory GNN
===================================================================
Derived from the D2STGNN architecture with ~20% algorithmic restructuring
for Walpurgis heterogeneous memory placement research.

Algorithmic changes vs D2STGNN:
  1. Exponential-decay layer aggregation replaces naive sum
  2. Adaptive skip-connection gating per decouple layer
  3. Embedding warmup: first N forward passes use identity projection
  4. Per-layer contribution tracking with EMA smoothing
  5. Full breakpoint-style debug infrastructure at every stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from collections import deque

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


# ═══════════ Debug Probing Infrastructure ═══════════ #

class TensorProbe:
    """Reusable tensor health inspector — drop-in replacement for manual prints.
    
    Usage at any breakpoint:
        probe = TensorProbe("layer3.attention")
        probe(some_tensor)        # prints shape, stats, anomaly flags
        probe.history()           # returns last N snapshots
    """
    
    _global_registry = {}  # all probes accessible by name
    
    def __init__(self, name, history_len=50, active=True):
        self.name = name
        self._active = active
        self._snapshots = deque(maxlen=history_len)
        self._call_count = 0
        TensorProbe._global_registry[name] = self
    
    def __call__(self, tensor, extra_tag=""):
        if not self._active or tensor is None:
            return tensor
        self._call_count += 1
        
        snapshot = {
            'call': self._call_count,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'mean': tensor.mean().item(),
            'std': tensor.std().item() if tensor.numel() > 1 else 0.0,
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'nan_count': torch.isnan(tensor).sum().item(),
            'inf_count': torch.isinf(tensor).sum().item(),
        }
        self._snapshots.append(snapshot)
        
        # Build anomaly flags
        flags = []
        if snapshot['nan_count'] > 0:
            flags.append(f"🔴NaN×{snapshot['nan_count']}")
        if snapshot['inf_count'] > 0:
            flags.append(f"🔴Inf×{snapshot['inf_count']}")
        if snapshot['std'] < 1e-7 and tensor.numel() > 1:
            flags.append("⚠️collapsed")
        if abs(snapshot['mean']) > 1e4:
            flags.append("⚠️large_mean")
        flag_str = " ".join(flags)
        
        tag = f" [{extra_tag}]" if extra_tag else ""
        print(f"    [PROBE] {self.name}{tag}: "
              f"shape={snapshot['shape']}, "
              f"μ={snapshot['mean']:+.5f}, σ={snapshot['std']:.5f}, "
              f"∈[{snapshot['min']:.5f}, {snapshot['max']:.5f}] "
              f"{flag_str}")
        return tensor
    
    def history(self):
        return list(self._snapshots)
    
    @staticmethod
    def dump_all():
        """Print summary of every registered probe — call at any debug breakpoint."""
        print(f"\n{'─'*70}")
        print(f"  TensorProbe Global Registry — {len(TensorProbe._global_registry)} probes")
        print(f"{'─'*70}")
        for name, probe in sorted(TensorProbe._global_registry.items()):
            last = probe._snapshots[-1] if probe._snapshots else None
            status = "no data" if last is None else (
                f"calls={probe._call_count}, last_mean={last['mean']:+.5f}")
            print(f"  {name:40s} | {status}")
        print(f"{'─'*70}\n")


# ═══════════ Decouple Layer ═══════════ #

class DecoupleLayer(nn.Module):
    """Separates diffusion (spatial) from inherent (temporal) signal pathways.
    
    Walpurgis changes vs upstream D2STGNN DecoupleLayer:
    - Added adaptive skip-connection gate (learnable α that blends residual)
    - Per-layer timing tracked with EMA instead of raw list
    - Probe objects at input/gate/output for breakpoint debugging
    """
    
    def __init__(self, hidden_dim, fk_dim=256, layer_idx=0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64
        )
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        
        # Walpurgis: learnable skip-gate — how much of the raw input leaks through
        # Initialized to 0.1 so early training relies mostly on the decouple path,
        # but the network can learn to bypass layers that contribute noise
        self.skip_gate = nn.Parameter(torch.tensor(0.1))
        
        # Timing: exponential moving average instead of unbounded list
        self._ema_forward_ms = 0.0
        self._ema_alpha = 0.05  # ~20-step window
        self._step_count = 0
        self._debug_on = True
        
        # Probes
        self._probe_in = TensorProbe(f"decouple_L{layer_idx}.input")
        self._probe_gated = TensorProbe(f"decouple_L{layer_idx}.gated")
        self._probe_out = TensorProbe(f"decouple_L{layer_idx}.output")
    
    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        t0 = time.perf_counter()
        
        if self._debug_on:
            self._probe_in(history_data)
        
        gated_history = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data
        )
        
        if self._debug_on:
            self._probe_gated(gated_history)
        
        dif_backcast, dif_forecast = self.dif_layer(
            history_data=history_data, gated_history_data=gated_history,
            dynamic_graph=dynamic_graph, static_graph=static_graph
        )
        
        inh_backcast, inh_forecast = self.inh_layer(dif_backcast)
        
        # Walpurgis: adaptive skip connection
        # α·(raw input) + (1-α)·(decouple output)
        # clamp α to [0, 0.5] so decouple path always dominates
        alpha = torch.clamp(torch.sigmoid(self.skip_gate), 0.0, 0.5)
        
        # Match sequence lengths for skip: take the shorter one
        skip_len = min(history_data.shape[1], inh_backcast.shape[1])
        blended = alpha * history_data[:, :skip_len, :, :] + (1.0 - alpha) * inh_backcast[:, :skip_len, :, :]
        
        # If inh_backcast is shorter (due to temporal pooling), pad back
        if inh_backcast.shape[1] > skip_len:
            blended = torch.cat([blended, inh_backcast[:, skip_len:, :, :]], dim=1)
        
        # Timing EMA
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._ema_forward_ms = (self._ema_alpha * elapsed_ms 
                                + (1.0 - self._ema_alpha) * self._ema_forward_ms)
        self._step_count += 1
        
        if self._debug_on and self._step_count % 100 == 0:
            tier_label = ("HBM" if self._ema_forward_ms > 5.0 
                          else "GDDR" if self._ema_forward_ms > 1.0 
                          else "DRAM")
            print(f"    [TIER] Layer[{self.layer_idx}] ema={self._ema_forward_ms:.2f}ms "
                  f"→ {tier_label} | skip_α={alpha.item():.4f}")
        
        if self._debug_on:
            self._probe_out(blended)
        
        return blended, dif_forecast, inh_forecast


# ═══════════ Main Model ═══════════ #

class D2STGNN(nn.Module):
    """Decoupled Spatial-Temporal Graph Neural Network — Walpurgis variant.
    
    Architecture is structurally equivalent to upstream D2STGNN with these
    algorithmic differences:
    
    1. **Exponential-decay aggregation**: layer forecasts are combined with
       weights w_i = exp(-λ·i) instead of uniform sum. λ is a learnable
       scalar initialized at 0.3. This lets the network learn whether to
       trust shallow or deep layers more.
    
    2. **Embedding warmup**: for the first `warmup_steps` forward calls,
       the input embedding uses a scaled identity projection instead of
       the learned linear. This prevents early random embeddings from
       dominating gradient signals before the graph structure has been
       learned.
    
    3. **Debug infrastructure**: TensorProbe at every stage, comprehensive
       per-layer contribution tracking, and a `snapshot()` method that
       dumps the entire model state as a JSON-serializable dict.
    """
    
    def __init__(self, **model_args):
        super().__init__()
        # ── Architecture dimensions ── #
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._forecast_dim  = 256
        self._output_hidden = 512
        self._output_dim    = model_args['seq_length']
        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 5
        
        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True
        self._model_args = model_args

        # ── Input embedding ── #
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # ── Temporal embeddings ── #
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # ── Decouple layers ── #
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                          layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])

        # ── Dynamic graph constructor ── #
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        
        # ── Node embeddings ── #
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # ── Output head ── #
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # ── Walpurgis: learnable aggregation decay ── #
        # λ controls how fast layer contributions decay: w_i = exp(-λ·i)
        # Higher λ = trust shallow layers more; lower λ = more uniform
        self._agg_lambda = nn.Parameter(torch.tensor(0.3))
        
        self._init_parameters()
        
        # ── Debug state ── #
        self._debug_on = True
        self._fwd_count = 0
        self._warmup_steps = 200  # use identity embedding for first N steps
        self._probe_embed = TensorProbe("model.embed_output")
        self._probe_final = TensorProbe("model.final_prediction")
        self._contribution_ema = {}  # layer_idx → EMA of contribution norm
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def toggle_debug(self, on: bool):
        """Master switch for all debug output in the model hierarchy."""
        self._debug_on = on
        for layer in self.layers:
            layer._debug_on = on

    def _build_graphs(self, **inputs):
        """Construct static + dynamic graphs from node embeddings."""
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        
        static_graph = []
        if self._model_args['sta_graph']:
            # Softmax-normalized attention graph
            attn = F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)
            static_graph.append(attn)
        
        dynamic_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        
        return static_graph, dynamic_graph

    def _extract_temporal_features(self, history_data):
        """Separate raw features from temporal indices and look up embeddings."""
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        
        # Temporal index extraction
        time_of_day_idx = (history_data[:, :, :, num_feat] * 288).long()
        day_of_week_idx = history_data[:, :, :, num_feat + 1].long()
        
        time_feat = self.T_i_D_emb[time_of_day_idx]
        day_feat = self.D_i_W_emb[day_of_week_idx]
        
        # Strip temporal indices from input
        raw_data = history_data[:, :, :, :num_feat]
        return raw_data, node_emb_u, node_emb_d, time_feat, day_feat

    def forward(self, history_data):
        """
        Forward pass with full debug instrumentation.
        
        Input:  [B, L, N, C]  — batch, seq_len, num_nodes, channels
        Output: [B, N, L']    — batch, num_nodes, prediction_horizon
        
        To inspect state at any point, set a Python breakpoint and call:
            TensorProbe.dump_all()
            self.snapshot()
        """
        self._fwd_count += 1
        t_wall = time.perf_counter()
        verbose = self._debug_on and (self._fwd_count % 200 == 1)
        
        if verbose:
            print(f"\n  ┌─ FWD #{self._fwd_count} ─ input: {list(history_data.shape)}")

        # ── Step 1: Feature extraction ── #
        t0 = time.perf_counter()
        raw_data, node_emb_u, node_emb_d, time_feat, day_feat = \
            self._extract_temporal_features(history_data)
        if verbose:
            print(f"  │ feature_extract: {(time.perf_counter()-t0)*1000:.1f}ms")

        # ── Step 2: Graph construction ── #
        t0 = time.perf_counter()
        static_graph, dynamic_graph = self._build_graphs(
            node_embedding_u=node_emb_u, node_embedding_d=node_emb_d,
            history_data=raw_data, time_in_day_feat=time_feat,
            day_in_week_feat=day_feat
        )
        if verbose:
            print(f"  │ graph_build: {(time.perf_counter()-t0)*1000:.1f}ms, "
                  f"static={len(static_graph)} dynamic={len(dynamic_graph)}")

        # ── Step 3: Embedding (with warmup bypass) ── #
        if self._fwd_count <= self._warmup_steps:
            # Warmup: scaled identity — just zero-pad or truncate to hidden_dim
            # This prevents random embedding weights from dominating early gradients
            if self._in_feat <= self._hidden_dim:
                embedded = F.pad(raw_data, (0, self._hidden_dim - self._in_feat))
            else:
                embedded = raw_data[:, :, :, :self._hidden_dim]
            embedded = embedded * 0.1  # scale down to match typical embedding magnitude
        else:
            embedded = self.embedding(raw_data)
        
        if verbose:
            self._probe_embed(embedded, f"warmup={'yes' if self._fwd_count<=self._warmup_steps else 'no'}")

        # ── Step 4: Decouple layers with contribution tracking ── #
        dif_forecasts = []
        inh_forecasts = []
        residual = embedded
        layer_stats = []
        
        for i, layer in enumerate(self.layers):
            t0 = time.perf_counter()
            residual, dif_fc, inh_fc = layer(
                residual, dynamic_graph, static_graph,
                node_emb_u, node_emb_d, time_feat, day_feat
            )
            layer_ms = (time.perf_counter() - t0) * 1000
            
            # Track contribution magnitudes with EMA
            dif_norm = dif_fc.norm().item()
            inh_norm = inh_fc.norm().item()
            
            if i not in self._contribution_ema:
                self._contribution_ema[i] = {'dif': dif_norm, 'inh': inh_norm}
            else:
                α = 0.1
                self._contribution_ema[i]['dif'] = α * dif_norm + (1-α) * self._contribution_ema[i]['dif']
                self._contribution_ema[i]['inh'] = α * inh_norm + (1-α) * self._contribution_ema[i]['inh']
            
            layer_stats.append({
                'idx': i, 'ms': layer_ms,
                'dif': dif_norm, 'inh': inh_norm,
                'skip_α': layer.skip_gate.item()
            })
            
            if verbose:
                print(f"  │ Layer {i}: {layer_ms:.1f}ms, "
                      f"dif_norm={dif_norm:.2f}, inh_norm={inh_norm:.2f}, "
                      f"skip_α={layer.skip_gate.item():.4f}")
            
            dif_forecasts.append(dif_fc)
            inh_forecasts.append(inh_fc)

        # ── Step 5: Exponential-decay aggregation ── #
        # w_i = exp(-λ·i), normalized to sum to 1
        # Upstream uses uniform sum; this lets the network learn layer importance
        lam = torch.clamp(self._agg_lambda, min=0.01, max=2.0)
        raw_weights = [torch.exp(-lam * i) for i in range(self._num_layers)]
        weight_total = sum(w.item() for w in raw_weights)
        
        agg_dif = sum(
            (w / weight_total) * h for w, h in zip(raw_weights, dif_forecasts)
        )
        agg_inh = sum(
            (w / weight_total) * h for w, h in zip(raw_weights, inh_forecasts)
        )
        combined = agg_dif + agg_inh
        
        # ── Step 6: Output projection ── #
        prediction = self.out_fc_2(F.relu(self.out_fc_1(F.relu(combined))))
        prediction = prediction.transpose(1, 2).contiguous().view(
            prediction.shape[0], prediction.shape[2], -1
        )
        
        if verbose:
            self._probe_final(prediction)
            norm_w = [f"{(w/weight_total).item():.3f}" for w in raw_weights]
            print(f"  │ agg_weights: {norm_w} (λ={lam.item():.4f})")
            
            # Contribution analysis
            total_dif = sum(s['dif'] for s in layer_stats) + 1e-8
            total_inh = sum(s['inh'] for s in layer_stats) + 1e-8
            for s in layer_stats:
                print(f"  │   L{s['idx']}: dif={s['dif']/total_dif*100:.1f}% "
                      f"inh={s['inh']/total_inh*100:.1f}% "
                      f"time={s['ms']:.1f}ms")
            print(f"  └─ total: {(time.perf_counter()-t_wall)*1000:.1f}ms\n")

        return prediction

    def snapshot(self):
        """Dump full model state as dict — call from debugger or test harness.
        
        Returns a dict with shapes, parameter stats, gradient norms, and
        contribution EMAs. Designed for JSON serialization or pdb inspection.
        """
        state = {
            'fwd_count': self._fwd_count,
            'agg_lambda': self._agg_lambda.item(),
            'contribution_ema': dict(self._contribution_ema),
            'parameters': {},
        }
        for name, p in self.named_parameters():
            entry = {
                'shape': list(p.shape),
                'mean': p.data.mean().item(),
                'std': p.data.std().item() if p.numel() > 1 else 0.0,
                'norm': p.data.norm().item(),
            }
            if p.grad is not None:
                entry['grad_norm'] = p.grad.data.norm().item()
            state['parameters'][name] = entry
        return state
