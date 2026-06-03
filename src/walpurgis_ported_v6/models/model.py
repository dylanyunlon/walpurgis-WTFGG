"""Main model — with highway embedding gate and output residual.

Changes
-------
1. Input embedding: adds a highway gate that lets a fraction of the raw
   input bypass the linear projection.  This preserves the original
   traffic signal magnitude in the hidden space, which helps the forecast
   branch produce well-scaled outputs without relying entirely on the
   scaler inverse.
2. Output layer: adds a residual connection from the *input* embedding
   (time-averaged) directly to the final forecast.  This gives the model
   a direct gradient path from loss to input, bypassing all 5 decouple
   layers — critical for avoiding vanishing gradients.
3. Every major stage in ``forward`` has a diagnostic checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from walpurgis_ported_v6 import _dbg


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim,
                                  **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim,
                                  **model_args)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(
            node_emb_u, node_emb_d,
            time_in_day_feat, day_in_week_feat, history_data)
        dif_res, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_res, inh_fk = self.inh_layer(dif_res)
        return inh_res, dif_fk, inh_fk


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self._in_feat = model_args['num_feat']
        self._hidden_dim = model_args['num_hidden']
        self._node_dim = model_args['node_hidden']
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = model_args['seq_length']
        self._num_nodes = model_args['num_nodes']
        self._k_s = model_args['k_s']
        self._k_t = model_args['k_t']
        self._num_layers = 5
        self._gap = model_args['gap']

        model_args['use_pre'] = False
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True
        self._model_args = model_args

        # ── highway embedding gate ──
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.highway_gate = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # decouple layers
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                          **model_args)
            for _ in range(self._num_layers)
        ])

        # dynamic graph
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # output — with residual shortcut
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, self._gap)
        # residual projection: hidden_dim → forecast_dim
        self.skip_proj = nn.Linear(self._hidden_dim, self._forecast_dim)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(
                F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        nf = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        tid = self.T_i_D_emb[
            (history_data[:, :, :, nf] * 288).type(torch.LongTensor)]
        diw = self.D_i_W_emb[
            (history_data[:, :, :, nf + 1]).type(torch.LongTensor)]
        raw = history_data[:, :, :, :nf]
        return raw, node_emb_u, node_emb_d, tid, diw

    def forward(self, history_data):
        # ── 1. Prepare ──
        raw, node_u, node_d, tid, diw = self._prepare_inputs(history_data)
        _dbg("Model.input", raw)

        # ── 2. Graphs ──
        static_g, dynamic_g = self._graph_constructor(
            node_embedding_u=node_u, node_embedding_d=node_d,
            history_data=raw, time_in_day_feat=tid,
            day_in_week_feat=diw)
        _dbg("Model.static_graph", static_g[0] if static_g else None)

        # ── 3. Highway embedding ──
        h_proj = self.embedding(raw)
        gate = torch.sigmoid(self.highway_gate(raw))
        # gate blends projected features with zero-padded raw bypass
        raw_pad = F.pad(raw, (0, self._hidden_dim - self._in_feat))
        h = gate * h_proj + (1 - gate) * raw_pad
        _dbg("Model.highway_emb", h, gate_mean=gate.mean().item())

        # save for output residual
        emb_mean = h.mean(dim=1)   # (B, N, D)

        dif_fk_list = []
        inh_fk_list = []
        signal = h

        # ── 4. Decouple stack ──
        for i, layer in enumerate(self.layers):
            signal, dif_fk, inh_fk = layer(
                signal, dynamic_g, static_g,
                node_u, node_d, tid, diw)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)
            _dbg(f"Model.layer_{i}", signal)

        # ── 5. Output ──
        dif_agg = sum(dif_fk_list)
        inh_agg = sum(inh_fk_list)
        forecast_hidden = dif_agg + inh_agg

        # residual shortcut from embedding
        skip = self.skip_proj(emb_mean).unsqueeze(1)
        forecast_hidden = forecast_hidden + skip
        _dbg("Model.forecast_hidden", forecast_hidden)

        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _dbg("Model.output", forecast)

        return forecast
