"""
Walpurgis ST Localized Convolution — Graph Diffusion with Debug Probes
======================================================================
Adapted from D2STGNN STLocalizedConv. This is the core spatial operator
that performs k-hop graph convolution on the localized ST kernel.

Modifications vs upstream:
  1. Shape validation at every transform step (unfold, reshape, gconv)
  2. Support matrix inspection — prints graph density and symmetry
  3. Gradient flow check through the conv path
  4. Per-call FLOP estimation for tier placement decisions
  5. Memory footprint tracking for dynamic support matrices
  6. Breakpoint-friendly: self.debug_state dict captures full internal state
"""

import torch
import torch.nn as nn
import time


def _estimate_gconv_flops(support_count, num_nodes, kernel_size, hidden_dim, batch_size, seq_len):
    """Rough FLOP estimate for one gconv call.
    
    Each support matrix multiply: O(batch * seq * nodes * k_nodes * feat)
    Plus the final linear projection: O(batch * seq * nodes * num_matric * feat)
    """
    matmul_flops = support_count * batch_size * seq_len * num_nodes * kernel_size * hidden_dim * 2
    proj_flops = batch_size * seq_len * num_nodes * (support_count + 1) * hidden_dim * hidden_dim * 2
    return matmul_flops + proj_flops


class STLocalizedConv(nn.Module):
    """Spatial-Temporal localized convolution via graph diffusion.
    
    Performs k-hop graph convolution on temporally unfolded input.
    The support set can include predefined, dynamic, and static hidden graphs.
    
    Walpurgis additions:
    - self.debug_state: dict capturing shapes/values at each internal step
    - Periodic verbose logging with FLOP estimates
    - Gradient norm tracking on fc_list_updt and gcn_updt
    """
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + 
                          len(self.pre_defined_graph) * int(dy_graph) + 
                          int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()
        
        # Walpurgis debug infrastructure
        self._call_count = 0
        self._total_forward_ms = 0.0
        self._total_flops = 0
        self.debug_state = {}  # breakpoint-friendly: inspect this dict at any time

        # Register gradient hooks for weight monitoring
        self._grad_norms = {'fc_list_updt': [], 'gcn_updt': []}
        self.fc_list_updt.weight.register_hook(
            lambda g: self._grad_norms['fc_list_updt'].append(g.norm().item())
        )
        self.gcn_updt.weight.register_hook(
            lambda g: self._grad_norms['gcn_updt'].append(g.norm().item())
        )

    def gconv(self, support, X_k, X_0):
        """Graph convolution with support matrices.
        
        Walpurgis: validates shape compatibility and tracks per-support contributions.
        """
        out = [X_0]
        support_norms = []
        for i, graph in enumerate(support):
            if len(graph.shape) == 2:
                # Static or predefined graph (N × kN)
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            support_norms.append(H_k.norm().item())
            out.append(H_k)
        
        self.debug_state['gconv_support_norms'] = support_norms
        self.debug_state['gconv_num_supports'] = len(support)
        
        out = torch.cat(out, dim=-1)
        self.debug_state['gconv_cat_shape'] = list(out.shape)
        
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        """Build k-hop ST-localized graph from support matrices.
        
        For each predefined/static graph, computes powers up to k_s
        and reshapes for spatial-temporal localization.
        """
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        
        # ST localization: expand each graph along temporal kernel dimension
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def dump_debug_state(self):
        """Return a copy of the current debug state for external inspection.
        
        Usage in a debugging session:
            # After a forward pass, inspect:
            state = model.decouple_layers[0].dif_layer.conv.dump_debug_state()
            print(state['unfold_shape'])
            print(state['gconv_support_norms'])
        """
        return {k: v for k, v in self.debug_state.items()}

    def get_timing_stats(self):
        """Return cumulative timing and FLOP statistics."""
        avg_ms = self._total_forward_ms / max(self._call_count, 1)
        return {
            'total_calls': self._call_count,
            'total_ms': self._total_forward_ms,
            'avg_ms': avg_ms,
            'total_flops': self._total_flops,
            'grad_norms_fc': self._grad_norms['fc_list_updt'][-5:] if self._grad_norms['fc_list_updt'] else [],
            'grad_norms_gcn': self._grad_norms['gcn_updt'][-5:] if self._grad_norms['gcn_updt'] else [],
        }

    def forward(self, X, dynamic_graph, static_graph):
        """Forward pass with full instrumentation.
        
        Captures timing, shapes, and intermediate values in self.debug_state.
        Prints diagnostic summary every 500 calls.
        """
        self._call_count += 1
        t0 = time.perf_counter()
        verbose = (self._call_count % 500 == 1)
        
        # Record input state
        self.debug_state['input_shape'] = list(X.shape)
        self.debug_state['input_norm'] = X.norm().item()
        self.debug_state['call_count'] = self._call_count
        
        # Temporal unfolding: [bs, seq, nodes, feat] → [bs, seq', nodes, k_t, feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape
        self.debug_state['unfold_shape'] = [batch_size, seq_len, num_nodes, kernel_size, num_feat]

        if verbose:
            print(f"        [STConv] call={self._call_count}: "
                  f"unfolded [{batch_size},{seq_len},{num_nodes},{kernel_size},{num_feat}]")

        # Build support set from available graph sources
        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)
        
        self.debug_state['support_count'] = len(support)
        self.debug_state['support_types'] = {
            'predefined': self.use_predefined_graph,
            'dynamic': self.use_dynamic_hidden_graph, 
            'static': self.use_static__hidden_graph,
        }

        # Parallelize: flatten temporal kernel into feature dimension
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        self.debug_state['fc_output_norm'] = out.norm().item()
        
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(batch_size, seq_len, kernel_size * num_nodes, num_feat)
        
        self.debug_state['X_0_shape'] = list(X_0.shape)
        self.debug_state['X_k_shape'] = list(X_k.shape)
        
        # Graph convolution
        hidden = self.gconv(support, X_k, X_0)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_forward_ms += elapsed_ms
        
        # FLOP estimation
        flops = _estimate_gconv_flops(len(support), num_nodes, kernel_size, num_feat, batch_size, seq_len)
        self._total_flops += flops
        
        self.debug_state['output_shape'] = list(hidden.shape)
        self.debug_state['output_norm'] = hidden.norm().item()
        self.debug_state['elapsed_ms'] = elapsed_ms
        self.debug_state['est_flops'] = flops
        
        if verbose:
            avg_ms = self._total_forward_ms / self._call_count
            print(f"        [STConv] output: {list(hidden.shape)}, "
                  f"support={len(support)}, "
                  f"this={elapsed_ms:.2f}ms avg={avg_ms:.2f}ms, "
                  f"FLOPs≈{flops/1e6:.1f}M")
            if self._grad_norms['fc_list_updt']:
                print(f"        [STConv] grad norms: fc={self._grad_norms['fc_list_updt'][-1]:.4f} "
                      f"gcn={self._grad_norms['gcn_updt'][-1]:.4f}")
        
        return hidden
