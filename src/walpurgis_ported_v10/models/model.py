import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_TAG = "model"


def _mish(x):
    return x * torch.tanh(F.softplus(x))


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

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

        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # 改动4: highway gate — σ(W·x) 控制多少原始特征直接透传
        # upstream 直接 embedding(x), 无 gate
        self.highway_fc = nn.Linear(self._in_feat, self._hidden_dim)
        self.highway_proj = nn.Linear(self._in_feat, self._hidden_dim)

        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                           **model_args)
             for _ in range(self._num_layers)])

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # 改动1: 输出层 ReLU → Mish
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # 改动2: 可学习层权重 — upstream 用 sum
        self.layer_logits_dif = nn.Parameter(torch.zeros(self._num_layers))
        self.layer_logits_inh = nn.Parameter(torch.zeros(self._num_layers))
        self.log_agg_tau = nn.Parameter(torch.zeros(1))  # 聚合温度

        # 改动3: static graph 温度
        self.log_static_tau = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        # 改动5: kaiming init 替代 xavier
        nn.init.kaiming_uniform_(self.T_i_D_emb)
        nn.init.kaiming_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            # 改动3: 温度缩放 softmax
            tau_s = torch.exp(self.log_static_tau).clamp(min=0.1, max=10.0)
            raw = torch.mm(E_d, E_u.T) / tau_s
            static_graph = [F.softmax(F.relu(raw), dim=1)]
            _dbg(_TAG, "static_graph", tau_s=tau_s, graph=static_graph[0])
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
        history_data = history_data[:, :, :, :nf]
        return history_data, node_emb_u, node_emb_d, tid, diw

    def forward(self, history_data):
        (history_data, node_emb_u, node_emb_d,
         tid, diw) = self._prepare_inputs(history_data)

        _dbg(_TAG, "input", history=history_data, tid=tid, diw=diw)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_emb_u, node_embedding_d=node_emb_d,
            history_data=history_data,
            time_in_day_feat=tid, day_in_week_feat=diw)

        # 改动4: highway gate
        gate = torch.sigmoid(self.highway_fc(history_data))
        embed = self.embedding(history_data)
        proj = self.highway_proj(history_data)
        history_data = gate * embed + (1.0 - gate) * proj

        _dbg(_TAG, "highway", gate_mean=gate.mean(), embed=embed)

        dif_fk_list = []
        inh_fk_list = []
        seq = history_data
        for i, layer in enumerate(self.layers):
            seq, dif_fk, inh_fk = layer(
                seq, dynamic_graph, static_graph,
                node_emb_u, node_emb_d, tid, diw)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)
            _dbg(_TAG, f"layer_{i}", seq=seq, dif_fk=dif_fk, inh_fk=inh_fk)

        # 改动2: softmax 层权重聚合
        agg_tau = torch.exp(self.log_agg_tau).clamp(min=0.1, max=5.0)
        w_dif = F.softmax(self.layer_logits_dif / agg_tau, dim=0)
        w_inh = F.softmax(self.layer_logits_inh / agg_tau, dim=0)

        dif_fk_agg = sum(w * fk for w, fk in zip(w_dif, dif_fk_list))
        inh_fk_agg = sum(w * fk for w, fk in zip(w_inh, inh_fk_list))
        forecast_hidden = dif_fk_agg + inh_fk_agg

        _dbg(_TAG, "aggregation",
             w_dif=w_dif, w_inh=w_inh, agg_tau=agg_tau,
             forecast=forecast_hidden)

        # 改动1: Mish 输出激活
        forecast = self.out_fc_2(_mish(self.out_fc_1(_mish(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        _dbg(_TAG, "output", forecast=forecast)
        return forecast
