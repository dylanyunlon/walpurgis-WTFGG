"""D2STGNN Nebula: DenseNet-style concat layer aggregation, Maxout output network,
Fourier feature PE, IndRNN + flash attention, hyperbolic distance graphs."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


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
        if _NEB_DBG:
            de = dif_fk.detach().norm().item(); ie = inh_fk.detach().norm().item()
            print(f"[NEB:decouple@model] dif_energy={de:.4f} inh_energy={ie:.4f} ratio={de/max(ie,1e-8):.4f}", file=sys.stderr)
        return inh_res, dif_fk, inh_fk


class MaxoutNetwork(nn.Module):
    """Maxout network: each output unit is the max over k affine pieces.
    f(x) = max_k (W_k * x + b_k).
    Replaces ReLU(FC)->FC output with piecewise linear maxout for richer function class."""
    def __init__(self, in_dim, out_dim, num_pieces=4):
        super().__init__()
        self.num_pieces = num_pieces
        self.out_dim = out_dim
        # k linear pieces: each maps in_dim -> out_dim
        self.linear = nn.Linear(in_dim, out_dim * num_pieces)

    def forward(self, x):
        """x: [..., in_dim] -> [..., out_dim]"""
        pieces = self.linear(x)  # [..., out_dim * num_pieces]
        shape = list(pieces.shape[:-1]) + [self.out_dim, self.num_pieces]
        pieces = pieces.view(*shape)  # [..., out_dim, num_pieces]
        out = pieces.max(dim=-1).values  # [..., out_dim]
        return out


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
        model_args['use_pre'] = False; model_args['dy_graph'] = True; model_args['sta_graph'] = True
        self._model_args = model_args
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))
        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args) for _ in range(self._num_layers)])
        # Nebula: DenseNet-style aggregation — project concatenated layers to forecast dim
        # Each layer produces forecast_dim, concat all layers -> num_layers * forecast_dim
        self.dense_proj = nn.Linear(self._num_layers * self._forecast_dim, self._forecast_dim)
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        # Nebula: Maxout output network
        self.out_maxout_1 = MaxoutNetwork(self._forecast_dim, self._output_hidden, num_pieces=4)
        self.out_maxout_2 = MaxoutNetwork(self._output_hidden, model_args['gap'], num_pieces=4)
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
        # Nebula: DenseNet-style — collect and concatenate all layer outputs
        dif_concat_list, inh_concat_list = [], []
        inh_bc = history_data
        for idx, layer in enumerate(self.layers):
            inh_bc, dif_fk, inh_fk = layer(inh_bc, dynamic_graph, static_graph, neu, ned, tid, diw)
            dif_concat_list.append(dif_fk)
            inh_concat_list.append(inh_fk)
            if _NEB_DBG:
                print(f"[NEB:layer{idx}@model] backcast={list(inh_bc.shape)}", file=sys.stderr)
        # DenseNet aggregation: concatenate all layers along feature dim, then project
        dif_dense = torch.cat(dif_concat_list, dim=-1)  # [B, T', N, num_layers * fk_dim]
        inh_dense = torch.cat(inh_concat_list, dim=-1)
        dif_forecast = self.dense_proj(dif_dense)  # [B, T', N, fk_dim]
        inh_forecast = self.dense_proj(inh_dense)
        forecast_hidden = dif_forecast + inh_forecast
        # Nebula: Maxout output
        forecast = self.out_maxout_2(self.out_maxout_1(forecast_hidden))
        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        if _NEB_DBG:
            print(f"[NEB:output@model] forecast={list(forecast.shape)} range=[{forecast.min().item():.4f},{forecast.max().item():.4f}]", file=sys.stderr)
        return forecast
