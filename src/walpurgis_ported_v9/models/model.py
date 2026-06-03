"""
model.py — v9 port (D2STGNN)
Algo delta:
  1. embedding 后接 highway gate: g=σ(FC(x)), out = g*embed + (1-g)*x_proj
     让模型决定多少原始特征直接透传
  2. 输出层 ReLU → LeakyReLU(0.1), 允许负值梯度流过
  3. _graph_constructor: static_graph softmax 前加可学习温度 T (init=1.0)
  4. 各层 forecast hidden 汇总时加可学习层权重 w_l (softmax归一化),
     而非简单 sum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from walpurgis_ported_v9 import _dbg

_TAG = "model"


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

        model_args['use_pre'] = False
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True
        self._model_args = model_args

        # embedding + v9: highway gate
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.hw_gate_fc = nn.Linear(self._in_feat, self._hidden_dim)
        self.hw_proj = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # layers
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)
             for _ in range(self._num_layers)])

        # v9: learnable layer weights for forecast aggregation
        self.layer_weight_dif = nn.Parameter(torch.ones(self._num_layers))
        self.layer_weight_inh = nn.Parameter(torch.ones(self._num_layers))

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # v9: LeakyReLU output
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
        self.out_act = nn.LeakyReLU(0.1)

        # v9: static graph temperature
        self.static_temp = nn.Parameter(torch.ones(1))

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
            # v9: temperature-scaled softmax
            T = torch.clamp(self.static_temp, min=0.01)
            logits = torch.mm(E_d, E_u.T) / T
            static_graph = [F.softmax(F.relu(logits), dim=1)]
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
        tid = self.T_i_D_emb[(history_data[:, :, :, nf] * 288).type(torch.LongTensor)]
        diw = self.D_i_W_emb[(history_data[:, :, :, nf + 1]).type(torch.LongTensor)]
        history_data = history_data[:, :, :, :nf]
        return history_data, node_emb_u, node_emb_d, tid, diw

    def forward(self, history_data):
        history_data, node_emb_u, node_emb_d, tid, diw = self._prepare_inputs(history_data)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_emb_u, node_embedding_d=node_emb_d,
            history_data=history_data, time_in_day_feat=tid, day_in_week_feat=diw)

        # v9: highway-gated embedding
        emb = self.embedding(history_data)
        gate = torch.sigmoid(self.hw_gate_fc(history_data))
        proj = self.hw_proj(history_data)
        history_data = gate * emb + (1.0 - gate) * proj

        _dbg(_TAG, f"hw_gate∈[{gate.min().item():.3f},{gate.max().item():.3f}]")

        dif_fk_list = []
        inh_fk_list = []
        inh_res = history_data
        for _, layer in enumerate(self.layers):
            inh_res, dif_fk, inh_fk = layer(
                inh_res, dynamic_graph, static_graph,
                node_emb_u, node_emb_d, tid, diw)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)

        # v9: weighted layer aggregation
        w_dif = F.softmax(self.layer_weight_dif, dim=0)
        w_inh = F.softmax(self.layer_weight_inh, dim=0)
        dif_fk = sum(w * h for w, h in zip(w_dif, dif_fk_list))
        inh_fk = sum(w * h for w, h in zip(w_inh, inh_fk_list))
        forecast_hidden = dif_fk + inh_fk

        _dbg(_TAG, f"layer_w_dif={w_dif.data.tolist()}  layer_w_inh={w_inh.data.tolist()}")

        # v9: LeakyReLU output
        forecast = self.out_fc_2(self.out_act(self.out_fc_1(self.out_act(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        _dbg(_TAG, f"output  shape={list(forecast.shape)}  "
                    f"mean={forecast.mean().item():.4g}  std={forecast.std().item():.4g}")
        return forecast
