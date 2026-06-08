import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_reverie import _dbg, dataflow_checkpoint

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_TAG = "model"


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
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)

        _dbg(f"{_TAG}/gated_data", gated_history_data, _TAG)

        dif_backcast_seq_res, dif_forecast_hidden = \
            self.dif_layer(
                history_data=history_data,
                gated_history_data=gated_history_data,
                dynamic_graph=dynamic_graph,
                static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = \
            self.inh_layer(dif_backcast_seq_res)
        return (inh_backcast_seq_res,
                dif_forecast_hidden, inh_forecast_hidden)


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

        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim,
                          fk_dim=self._forecast_dim, **model_args)
             for _ in range(self._num_layers)])

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)

        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # 改动8: LayerScale + GELU output head
        # LayerScale: 每层输出乘以可学习的小标量 (初始0.1), 稳定深层训练
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1)
            for _ in range(self._num_layers)
        ])

        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # 改动: GELU代替ReLU, 加LayerNorm稳定输出
        self.out_norm = nn.LayerNorm(self._output_hidden)

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
            static_graph = [
                F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
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
            (history_data[:, :, :, num_feat] * 288
             ).type(torch.LongTensor)]
        day_in_week_feat = self.D_i_W_emb[
            (history_data[:, :, :, num_feat + 1]
             ).type(torch.LongTensor)]
        history_data = history_data[:, :, :, :num_feat]
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        (history_data, node_embedding_u, node_embedding_d,
         time_in_day_feat, day_in_week_feat) = \
            self._prepare_inputs(history_data)

        dataflow_checkpoint("input", history_data,
                           expected_shape=[None, None, self._num_nodes, self._in_feat])

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
        for i, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, \
                inh_forecast_hidden = layer(
                    inh_backcast_seq_res, dynamic_graph,
                    static_graph, node_embedding_u,
                    node_embedding_d, time_in_day_feat,
                    day_in_week_feat)
            # 改动: LayerScale — 乘以可学习的小标量
            scale = self.layer_scales[i].abs()
            dif_forecast_hidden_list.append(
                dif_forecast_hidden * scale)
            inh_forecast_hidden_list.append(
                inh_forecast_hidden * scale)

            _dbg(f"{_TAG}/layer_{i}_scale", scale, _TAG)

        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        # 改动: GELU + LayerNorm output head
        h = F.gelu(self.out_fc_1(forecast_hidden))
        h = self.out_norm(h)
        forecast = self.out_fc_2(h)
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        _dbg(f"{_TAG}/forecast_final", forecast, _TAG)
        return forecast
