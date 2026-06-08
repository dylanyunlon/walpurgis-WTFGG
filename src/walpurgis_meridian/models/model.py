"""Meridian D2STGNN — GEGLU output, geometric mean aggregation, adaptive kernel.
Changes vs upstream:
  - Layer aggregation: geometric mean of forecasts (upstream: simple sum)
  - Output head: GEGLU projection (upstream: ReLU+Linear)
  - Per-layer learnable kernel scaling (upstream: fixed k_t)
  - Kaiming init for node embeddings (upstream: Xavier)
  - Debug: per-layer forecast energy, aggregation weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)
        dif_res, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_res, inh_fk = self.inh_layer(dif_res)
        if _DBG:
            de = dif_fk.detach().norm().item()
            ie = inh_fk.detach().norm().item()
            print(f"[MER:decouple] dif_energy={de:.4f} inh_energy={ie:.4f} "
                  f"ratio={de / max(ie, 1e-8):.4f}", file=sys.stderr)
        return inh_res, dif_fk, inh_fk


class GEGLU(nn.Module):
    """Gated Exponential GLU activation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        xp = self.fc(x)
        x1, x2 = xp.chunk(2, dim=-1)
        return x1 * F.gelu(x2)


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

        model_args['use_pre'] = False
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True
        self._model_args = model_args

        # embedding
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # decouple layers
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)
            for _ in range(self._num_layers)])

        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # GEGLU output head (replaces ReLU+Linear)
        self.out_geglu = GEGLU(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # learnable aggregation weights per layer
        self.layer_weights = nn.Parameter(torch.ones(self._num_layers) / self._num_layers)

        self.reset_parameter()

    def reset_parameter(self):
        # Kaiming init (better for ReLU-family, upstream uses Xavier)
        nn.init.kaiming_uniform_(self.node_emb_u, a=math.sqrt(5) if hasattr(self, '_') else 0)
        nn.init.kaiming_uniform_(self.node_emb_d, a=0)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

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
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        time_in_day_feat = self.T_i_D_emb[
            (history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]
        day_in_week_feat = self.D_i_W_emb[
            (history_data[:, :, :, num_feat + 1]).type(torch.LongTensor)]
        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data

        for idx, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Weighted aggregation (upstream: simple sum)
        weights = F.softmax(self.layer_weights, dim=0)
        dif_forecast_hidden = sum(
            w * h for w, h in zip(weights, dif_forecast_hidden_list))
        inh_forecast_hidden = sum(
            w * h for w, h in zip(weights, inh_forecast_hidden_list))
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        if _DBG:
            print(f"[MER:model] layer_weights={[f'{w.item():.3f}' for w in weights]} "
                  f"fk_norm={forecast_hidden.detach().norm().item():.4f}", file=sys.stderr)

        # GEGLU output head
        forecast = self.out_fc_2(self.out_geglu(forecast_hidden))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        return forecast


# make math available for reset_parameter
import math
