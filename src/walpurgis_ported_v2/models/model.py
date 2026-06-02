"""
D2STGNN — Decoupled Dynamic Spatial-Temporal Graph Neural Network.
Main model file: embedding → graph construction → N×DecoupleLayer → output FC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_DBG_MODEL = ("--debug-model" in sys.argv) or False
_LAYER_COUNT = 5     # number of stacked decouple layers


class DecoupleLayer(nn.Module):
    """Single decouple layer: Estimation Gate → DifBlock → InhBlock."""

    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64,
        )
        self.diffusion = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inherent  = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history, dyn_graph, sta_graph,
                node_emb_u, node_emb_d, tod_feat, dow_feat):
        """
        Returns
        -------
        residual_out     : [B, L', N, D]   — feed to next layer
        dif_fk_hidden    : [B, H, N, fk]   — diffusion forecast features
        inh_fk_hidden    : [B, H, N, fk]   — inherent forecast features
        """
        gated = self.gate(node_emb_u, node_emb_d, tod_feat, dow_feat, history)
        dif_residual, dif_fk = self.diffusion(
            history, gated, dyn_graph, sta_graph
        )
        inh_residual, inh_fk = self.inherent(dif_residual)
        return inh_residual, dif_fk, inh_fk


class D2STGNN(nn.Module):
    """
    Full model: input embedding → graph construction →
    stacked DecoupleLayer → sum forecast branches → regression head.
    """

    def __init__(self, **model_args):
        super().__init__()
        # ── dimensions ──
        self._d_in       = model_args['num_feat']
        self._d_hidden   = model_args['num_hidden']
        self._d_node     = model_args['node_hidden']
        self._d_forecast = 256
        self._d_out_fc   = 512
        self._horizon    = model_args['seq_length']

        self._n_nodes    = model_args['num_nodes']
        self._k_s        = model_args['k_s']
        self._k_t        = model_args['k_t']
        self._n_layers   = _LAYER_COUNT

        # hard-wire graph flags (consistent with original design)
        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True
        self._cfg = model_args

        # ── input embedding ──
        self.input_proj = nn.Linear(self._d_in, self._d_hidden)

        # ── temporal embeddings (learnable) ──
        self.time_of_day_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.day_of_week_emb = nn.Parameter(torch.empty(7,   model_args['time_emb_dim']))

        # ── decouple layers ──
        self.layers = nn.ModuleList([
            DecoupleLayer(self._d_hidden, fk_dim=self._d_forecast, **model_args)
            for _ in range(self._n_layers)
        ])

        # ── dynamic graph constructor ──
        if model_args['dy_graph']:
            self.dyn_graph_ctor = DynamicGraphConstructor(**model_args)

        # ── node embeddings ──
        self.node_emb_u = nn.Parameter(torch.empty(self._n_nodes, self._d_node))
        self.node_emb_d = nn.Parameter(torch.empty(self._n_nodes, self._d_node))

        # ── output regression head ──
        self.out_fc1 = nn.Linear(self._d_forecast, self._d_out_fc)
        self.out_fc2 = nn.Linear(self._d_out_fc, model_args['gap'])

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.time_of_day_emb)
        nn.init.xavier_uniform_(self.day_of_week_emb)

    # ──────────── graph construction ────────────

    def _build_graphs(self, **ctx):
        E_u = ctx['node_embedding_u']
        E_d = ctx['node_embedding_d']

        static = []
        if self._cfg['sta_graph']:
            # adaptive graph via softmax(ReLU(E_d · E_u^T))
            affinity = F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)
            static = [affinity]

        dynamic = []
        if self._cfg['dy_graph']:
            dynamic = self.dyn_graph_ctor(**ctx)

        if _DBG_MODEL:
            print(f"[DBG:model] _build_graphs  n_static={len(static)}  "
                  f"n_dynamic={len(dynamic)}")
        return static, dynamic

    # ──────────── input preparation ────────────

    def _prepare(self, raw_data):
        """Extract traffic signal, node embs, and time embs from raw input."""
        n_feat = self._cfg['num_feat']

        eu = self.node_emb_u
        ed = self.node_emb_d

        # temporal embedding look-up
        tod_idx = (raw_data[:, :, :, n_feat] * 288).long()
        dow_idx = raw_data[:, :, :, n_feat + 1].long()
        tod_feat = self.time_of_day_emb[tod_idx]    # [B, L, N, d_time]
        dow_feat = self.day_of_week_emb[dow_idx]    # [B, L, N, d_time]

        traffic = raw_data[:, :, :, :n_feat]

        if _DBG_MODEL:
            print(f"[DBG:model] _prepare  traffic={tuple(traffic.shape)}  "
                  f"tod_idx_range=[{tod_idx.min()},{tod_idx.max()}]  "
                  f"dow_idx_range=[{dow_idx.min()},{dow_idx.max()}]")
        return traffic, eu, ed, tod_feat, dow_feat

    # ──────────── forward pass ────────────

    def forward(self, history_data):
        """
        Parameters
        ----------
        history_data : [B, L, N, C]  — C includes traffic features + time indices

        Returns
        -------
        prediction : [B, N, horizon]
        """
        # step 1: prepare inputs
        traffic, eu, ed, tod, dow = self._prepare(history_data)

        # step 2: construct graphs
        sta_graph, dyn_graph = self._build_graphs(
            node_embedding_u=eu, node_embedding_d=ed,
            history_data=traffic, time_in_day_feat=tod, day_in_week_feat=dow,
        )

        # step 3: input embedding
        h = self.input_proj(traffic)

        if _DBG_MODEL:
            print(f"[DBG:model] after embed  h={tuple(h.shape)}  "
                  f"h_norm={h.norm().item():.4f}")

        # step 4: stacked decouple layers
        dif_forecasts = []
        inh_forecasts = []
        residual = h
        for i, layer in enumerate(self.layers):
            residual, dif_fk, inh_fk = layer(
                residual, dyn_graph, sta_graph, eu, ed, tod, dow
            )
            dif_forecasts.append(dif_fk)
            inh_forecasts.append(inh_fk)
            if _DBG_MODEL:
                print(f"[DBG:model] layer {i}  residual={tuple(residual.shape)}  "
                      f"dif_fk_norm={dif_fk.norm().item():.4f}  "
                      f"inh_fk_norm={inh_fk.norm().item():.4f}")

        # step 5: aggregate forecasts from all layers
        dif_agg = sum(dif_forecasts)
        inh_agg = sum(inh_forecasts)
        combined = dif_agg + inh_agg

        # step 6: regression head
        out = self.out_fc2(F.relu(self.out_fc1(F.relu(combined))))
        # reshape: [B, steps, N, gap] → [B, N, horizon]
        prediction = out.transpose(1, 2).contiguous().view(
            out.shape[0], out.shape[2], -1
        )

        if _DBG_MODEL:
            print(f"[DBG:model] output  shape={tuple(prediction.shape)}  "
                  f"range=[{prediction.min().item():.4g}, {prediction.max().item():.4g}]")
        return prediction
