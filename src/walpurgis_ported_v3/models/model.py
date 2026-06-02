"""
D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network.
Main model assembling decouple layers, graph constructors, and output heads.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_DBG = ("--debug-model" in sys.argv)


class DecoupleLayer(nn.Module):
    """One round of diffusion-inherent decoupling."""

    def __init__(self, hidden_dim, fk_dim=256, **kw):
        super().__init__()
        self.gate = EstimationGate(
            node_emb_dim=kw['node_hidden'],
            time_emb_dim=kw['time_emb_dim'],
            hidden_dim=64)
        self.dif  = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **kw)
        self.inh  = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **kw)

    def forward(self, hist, dyn_g, sta_g, emb_u, emb_d, t_day, t_week):
        gated = self.gate(emb_u, emb_d, t_day, t_week, hist)
        dif_res, dif_fk = self.dif(
            history_data=hist, gated_data=gated,
            dynamic_graph=dyn_g, static_graph=sta_g)
        inh_res, inh_fk = self.inh(dif_res)
        return inh_res, dif_fk, inh_fk


class D2STGNN(nn.Module):

    def __init__(self, **kw):
        super().__init__()
        # ── dimensions ──
        self._in_feat      = kw['num_feat']
        self._d_hidden     = kw['num_hidden']
        self._d_node       = kw['node_hidden']
        self._d_forecast   = 256
        self._d_out_hidden = 512
        self._horizon      = kw['seq_length']
        self._n_nodes      = kw['num_nodes']
        self._k_s          = kw['k_s']
        self._k_t          = kw['k_t']
        self._n_layers     = 5

        kw['use_pre']   = False
        kw['dy_graph']  = True
        kw['sta_graph'] = True
        self._kw = kw

        # ── input projection ──
        self.input_proj = nn.Linear(self._in_feat, self._d_hidden)

        # ── temporal embeddings ──
        self.T_i_D = nn.Parameter(torch.empty(288, kw['time_emb_dim']))
        self.D_i_W = nn.Parameter(torch.empty(7,   kw['time_emb_dim']))

        # ── decouple stack ──
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._d_hidden, fk_dim=self._d_forecast, **kw)
             for _ in range(self._n_layers)])

        # ── dynamic graph constructor ──
        if kw['dy_graph']:
            self.dyn_graph_ctor = DynamicGraphConstructor(**kw)

        # ── node embeddings ──
        self.emb_u = nn.Parameter(torch.empty(self._n_nodes, self._d_node))
        self.emb_d = nn.Parameter(torch.empty(self._n_nodes, self._d_node))

        # ── output MLP ──
        self.out_fc1 = nn.Linear(self._d_forecast, self._d_out_hidden)
        self.out_fc2 = nn.Linear(self._d_out_hidden, kw['gap'])

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.emb_u)
        nn.init.xavier_uniform_(self.emb_d)
        nn.init.xavier_uniform_(self.T_i_D)
        nn.init.xavier_uniform_(self.D_i_W)

    # ── graph construction ──

    def _build_graphs(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        sta = []
        if self._kw['sta_graph']:
            sta = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        dyn = []
        if self._kw['dy_graph']:
            dyn = self.dyn_graph_ctor(**inputs)
        return sta, dyn

    # ── input preparation ──

    def _split_inputs(self, raw):
        nf = self._kw['num_feat']
        emb_u = self.emb_u
        emb_d = self.emb_d
        t_day  = self.T_i_D[(raw[:, :, :, nf]   * 288).long()]
        t_week = self.D_i_W[(raw[:, :, :, nf+1]).long()]
        signal = raw[:, :, :, :nf]
        return signal, emb_u, emb_d, t_day, t_week

    # ── forward ──

    def forward(self, history_data):
        """
        history_data: (B, L, N, C)  where C = num_feat + 2 (time features).
        Returns: (B, N, horizon) predictions.
        """
        signal, eu, ed, t_day, t_week = self._split_inputs(history_data)

        if _DBG:
            print(f"[DBG:model] input  signal={tuple(signal.shape)}  "
                  f"t_day={tuple(t_day.shape)}  eu={tuple(eu.shape)}")

        sta_g, dyn_g = self._build_graphs(
            node_embedding_u=eu, node_embedding_d=ed,
            history_data=signal, time_in_day_feat=t_day,
            day_in_week_feat=t_week)

        # project to hidden dim
        h = self.input_proj(signal)

        dif_fks = []
        inh_fks = []
        residual = h
        for li, layer in enumerate(self.layers):
            residual, dif_fk, inh_fk = layer(
                residual, dyn_g, sta_g, eu, ed, t_day, t_week)
            dif_fks.append(dif_fk)
            inh_fks.append(inh_fk)
            if _DBG:
                print(f"[DBG:model] layer {li}  "
                      f"residual_norm={residual.norm().item():.2f}  "
                      f"dif_fk_mean={dif_fk.mean().item():.4f}")

        # ── output aggregation ──
        agg = sum(dif_fks) + sum(inh_fks)
        out = self.out_fc2(F.relu(self.out_fc1(F.relu(agg))))
        # reshape: (B, T/gap, N, gap) -> (B, N, T)
        out = out.transpose(1, 2).contiguous().view(
            out.shape[0], out.shape[2], -1)

        if _DBG:
            print(f"[DBG:model] output  shape={tuple(out.shape)}  "
                  f"range=[{out.min().item():.4f}, {out.max().item():.4f}]")

        return out
