"""D2STGNN Eclipse: EMA layer aggregation, GELU+SpectralNorm output, Gaussian noise embedding."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
from torch.nn.utils import spectral_norm
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)
        dif_res, dif_fk = self.dif_layer(history_data=history_data, gated_history_data=gated, dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_res, inh_fk = self.inh_layer(dif_res)
        if _ECL_DBG:
            de = dif_fk.detach().norm().item(); ie = inh_fk.detach().norm().item()
            print(f"[ECL:decouple@model] dif_energy={de:.4f} inh_energy={ie:.4f} ratio={de/max(ie,1e-8):.4f}", file=sys.stderr)
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
        self._noise_sigma = 0.01
        model_args['use_pre'] = False; model_args['dy_graph'] = True; model_args['sta_graph'] = True
        self._model_args = model_args
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))
        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args) for _ in range(self._num_layers)])
        self.log_ema_decay = nn.Parameter(torch.tensor(-1.0))
        if model_args['dy_graph']: self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.out_fc_1 = spectral_norm(nn.Linear(self._forecast_dim, self._output_hidden))
        self.out_fc_2 = spectral_norm(nn.Linear(self._output_hidden, model_args['gap']))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u); nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb); nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']; E_u = inputs['node_embedding_d']
        static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)] if self._model_args['sta_graph'] else []
        dynamic_graph = self.dynamic_graph_constructor(**inputs) if self._model_args['dy_graph'] else []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        nf = self._model_args['num_feat']
        t_idx = (history_data[:, :, :, nf] * 288).long().clamp(0, 287)
        d_idx = history_data[:, :, :, nf + 1].long().clamp(0, 6)
        return history_data[:, :, :, :nf], self.node_emb_u, self.node_emb_d, self.T_i_D_emb[t_idx], self.D_i_W_emb[d_idx]

    def forward(self, history_data):
        history_data, neu, ned, tid, diw = self._prepare_inputs(history_data)
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=neu, node_embedding_d=ned, history_data=history_data, time_in_day_feat=tid, day_in_week_feat=diw)
        history_data = self.embedding(history_data)
        if self.training:
            noise = torch.randn_like(history_data) * self._noise_sigma
            history_data = history_data + noise
            if _ECL_DBG: print(f"[ECL:noise@model] sigma={self._noise_sigma} noise_norm={noise.norm().item():.4f}", file=sys.stderr)
        ema_decay = torch.sigmoid(self.log_ema_decay)
        weights = [ema_decay ** (self._num_layers - 1 - l) for l in range(self._num_layers)]
        wsum = sum(weights)
        dif_list, inh_list = [], []
        inh_bc = history_data
        for idx, layer in enumerate(self.layers):
            inh_bc, dif_fk, inh_fk = layer(inh_bc, dynamic_graph, static_graph, neu, ned, tid, diw)
            dif_list.append(dif_fk * (weights[idx] / wsum))
            inh_list.append(inh_fk * (weights[idx] / wsum))
            if _ECL_DBG: print(f"[ECL:layer{idx}@model] backcast={list(inh_bc.shape)} w={weights[idx].item():.4f}", file=sys.stderr)
        forecast_hidden = sum(dif_list) + sum(inh_list)
        forecast = self.out_fc_2(F.gelu(self.out_fc_1(F.gelu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        if _ECL_DBG: print(f"[ECL:output@model] forecast={list(forecast.shape)} range=[{forecast.min().item():.4f},{forecast.max().item():.4f}]", file=sys.stderr)
        return forecast
