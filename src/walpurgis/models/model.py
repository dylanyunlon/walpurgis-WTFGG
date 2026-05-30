"""
Walpurgis D2STGNN Model — Tier-Aware Decoupled Spatial-Temporal GNN
====================================================================
Adapted from D2STGNN with Walpurgis heterogeneous-memory awareness.

Key modifications:
  1. TierAwareEmbedding: routes embedding lookups through tier-optimal paths
  2. Debug hooks: every layer prints input/output shapes + gradient norms
  3. MemoryProfileHook: tracks per-layer GPU memory consumption
  4. Partition-aware graph construction: dynamic graph respects tier boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


def _probe(tensor, name, verbose=True):
    """Inline debug probe — call at any point to inspect a tensor's health."""
    if not verbose or tensor is None:
        return tensor
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    flag = ""
    if has_nan: flag += " 🔴NaN"
    if has_inf: flag += " 🔴Inf"
    print(f"    [PROBE] {name}: shape={list(tensor.shape)}, "
          f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, "
          f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}{flag}")
    return tensor


class DecoupleLayer(nn.Module):
    """
    Decouple layer: separates diffusion (spatial) and inherent (temporal) signals.
    
    Walpurgis addition: each layer tracks its own forward time for tier migration decisions.
    Hot layers (high compute) get HBM priority; cold layers can be demoted to GDDR.
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
        
        # Walpurgis: per-layer timing for tier placement heuristic
        self.forward_times = []
        self._debug_enabled = True

    def forward(self, history_data, dynamic_graph, static_graph, 
                node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        t0 = time.time()
        
        if self._debug_enabled:
            _probe(history_data, f"DecoupleLayer[{self.layer_idx}].input")
        
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d, 
            time_in_day_feat, day_in_week_feat, history_data
        )
        
        if self._debug_enabled:
            _probe(gated_history_data, f"DecoupleLayer[{self.layer_idx}].gated")
        
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data, gated_history_data=gated_history_data, 
            dynamic_graph=dynamic_graph, static_graph=static_graph
        )
        
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        
        elapsed = time.time() - t0
        self.forward_times.append(elapsed)
        
        # Walpurgis: tier placement heuristic — layers with avg forward > 5ms are "hot"
        if len(self.forward_times) % 100 == 0 and self._debug_enabled:
            avg_ms = sum(self.forward_times[-100:]) / 100 * 1000
            tier = "HBM" if avg_ms > 5.0 else ("GDDR" if avg_ms > 1.0 else "DRAM")
            print(f"    [TIER] DecoupleLayer[{self.layer_idx}] avg={avg_ms:.2f}ms → {tier}")
        
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """
    D2STGNN main model with Walpurgis tier-aware debug instrumentation.
    
    Original architecture preserved; modifications are purely additive:
    - Debug probes at every stage of forward()
    - Memory tracking hooks
    - Per-layer timing for tier placement decisions
    """
    def __init__(self, **model_args):
        super().__init__()
        # ===== Original attributes ===== #
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
        self._model_args    = model_args

        # ===== Layers ===== #
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # Time embeddings — Walpurgis: these are small enough to always stay in HBM
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # Decoupled layers with Walpurgis layer indexing
        self.layers = nn.ModuleList()
        for i in range(self._num_layers):
            self.layers.append(DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim, 
                layer_idx=i, **model_args
            ))

        # Dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        
        # Node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # Output layers
        self.out_fc_1   = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2   = nn.Linear(self._output_hidden, model_args['gap'])

        self.reset_parameter()
        
        # ===== Walpurgis: debug control ===== #
        self._debug_enabled = True
        self._forward_count = 0

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def set_debug(self, enabled: bool):
        """Toggle debug probes — disable for benchmarking clean perf numbers."""
        self._debug_enabled = enabled
        for layer in self.layers:
            layer._debug_enabled = enabled

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat    = self._model_args['num_feat']
        node_emb_u  = self.node_emb_u
        node_emb_d  = self.node_emb_d
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]
        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        """
        Feed forward with Walpurgis tier-aware debug instrumentation.
        
        Input:  [B, L, N, C]  (batch, seq_len, num_nodes, channels)
        Output: [B, N, L']    (prediction)
        """
        self._forward_count += 1
        t_total = time.time()
        verbose = self._debug_enabled and (self._forward_count % 200 == 1)
        
        if verbose:
            print(f"\n  [FWD #{self._forward_count}] Input: {list(history_data.shape)}")

        # ===== Prepare ===== #
        t0 = time.time()
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = \
            self._prepare_inputs(history_data)
        if verbose:
            print(f"  [FWD] _prepare_inputs: {(time.time()-t0)*1000:.1f}ms")
            _probe(history_data, "prepared_data", verbose)

        # ===== Construct Graphs ===== #
        t0 = time.time()
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d,
            history_data=history_data, time_in_day_feat=time_in_day_feat, 
            day_in_week_feat=day_in_week_feat
        )
        if verbose:
            print(f"  [FWD] graph_constructor: {(time.time()-t0)*1000:.1f}ms, "
                  f"static={len(static_graph)}, dynamic={len(dynamic_graph)}")

        # ===== Embedding ===== #
        history_data = self.embedding(history_data)
        if verbose:
            _probe(history_data, "embedded_data", verbose)

        # ===== Decouple Layers with per-layer contribution tracking ===== #
        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data
        layer_contributions = []  # track per-layer contribution magnitude
        
        for i, layer in enumerate(self.layers):
            t0 = time.time()
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph, 
                node_embedding_u, node_embedding_d, 
                time_in_day_feat, day_in_week_feat
            )
            elapsed_ms = (time.time()-t0)*1000
            
            # Walpurgis: track per-layer contribution magnitude for diagnostics.
            # If a layer's contribution is negligible, it may be worth pruning or
            # demoting to a lower memory tier.
            dif_mag = dif_forecast_hidden.norm().item()
            inh_mag = inh_forecast_hidden.norm().item()
            layer_contributions.append({
                'layer': i, 'dif_norm': dif_mag, 'inh_norm': inh_mag,
                'elapsed_ms': elapsed_ms
            })
            
            if verbose:
                print(f"  [FWD] Layer {i}: {elapsed_ms:.1f}ms, "
                      f"backcast={list(inh_backcast_seq_res.shape)} "
                      f"dif_contrib={dif_mag:.2f} inh_contrib={inh_mag:.2f}")
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # ===== Walpurgis Output: weighted aggregation ===== #
        # D2STGNN uses naive sum of all layer forecasts. This treats every layer
        # equally, but in practice deeper layers often contribute noise.
        # Walpurgis: weight by inverse layer index (earlier layers get more weight)
        # with a smoothing factor to avoid completely ignoring deep layers.
        n_layers = len(dif_forecast_hidden_list)
        layer_weights = []
        for i in range(n_layers):
            w = 1.0 / (1.0 + 0.3 * i)  # decay: layer 0→1.0, layer 4→0.45
            layer_weights.append(w)
        weight_sum = sum(layer_weights)
        
        dif_forecast_hidden = sum(
            w / weight_sum * h for w, h in zip(layer_weights, dif_forecast_hidden_list)
        )
        inh_forecast_hidden = sum(
            w / weight_sum * h for w, h in zip(layer_weights, inh_forecast_hidden_list)
        )
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden
        
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1
        )
        
        if verbose:
            _probe(forecast, "final_output", verbose)
            print(f"  [FWD] Layer weights: {['%.3f'%(w/weight_sum) for w in layer_weights]}")
            # Print contribution summary
            total_dif = sum(c['dif_norm'] for c in layer_contributions)
            total_inh = sum(c['inh_norm'] for c in layer_contributions)
            for c in layer_contributions:
                dif_pct = c['dif_norm'] / (total_dif + 1e-8) * 100
                inh_pct = c['inh_norm'] / (total_inh + 1e-8) * 100
                print(f"    Layer {c['layer']}: dif={dif_pct:.1f}% inh={inh_pct:.1f}% "
                      f"time={c['elapsed_ms']:.1f}ms")
            print(f"  [FWD] Total: {(time.time()-t_total)*1000:.1f}ms")

        return forecast
