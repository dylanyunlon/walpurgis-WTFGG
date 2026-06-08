"""
D2STGNN — Penumbra变体
算法改写 (~20%):
  1. EMA衰减聚合: 各层forecast hidden按指数衰减加权
     越靠后的层权重越大, 衰减率可学习
  2. 嵌入后接Swish + 可学习偏差缩放
  3. 输出头: 双层FC + LayerNorm + 残差shortcut
  4. 集成所有子模块的算法改动
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import (_dbg, dataflow_checkpoint, dump_struct_state)


def _swish(x):
    return x * torch.sigmoid(x)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat,
            history_data)
        dif_backcast_seq_res, dif_forecast_hidden = \
            self.dif_layer(
                history_data=history_data,
                gated_history_data=gated_history_data,
                dynamic_graph=dynamic_graph,
                static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = \
            self.inh_layer(dif_backcast_seq_res)
        return (inh_backcast_seq_res,
                dif_forecast_hidden,
                inh_forecast_hidden)


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

        # 嵌入层 + Swish + 可学习缩放
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        self.embed_scale = nn.Parameter(
            torch.ones(self._hidden_dim))
        self.embed_bias = nn.Parameter(
            torch.zeros(self._hidden_dim))

        # 时间嵌入
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decouple层
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim,
                          fk_dim=self._forecast_dim,
                          **model_args)
        ])
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(self._hidden_dim,
                              fk_dim=self._forecast_dim,
                              **model_args))

        # EMA衰减聚合: 可学习的衰减率
        self.ema_decay = nn.Parameter(torch.tensor(0.8))

        # 动态图构造器
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)

        # 节点嵌入
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # 输出头: 双层FC + LayerNorm + 残差shortcut
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # 残差shortcut: 直接从forecast_dim投影到gap
        self.out_shortcut = nn.Linear(
            self._forecast_dim, model_args['gap'])

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
            dynamic_graph = \
                self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        t_idx = (history_data[:, :, :, num_feat] * 288
                 ).type(torch.LongTensor).clamp(0, 287)
        d_idx = (history_data[:, :, :, num_feat + 1]
                 ).type(torch.LongTensor).clamp(0, 6)
        time_in_day_feat = self.T_i_D_emb[t_idx]
        day_in_week_feat = self.D_i_W_emb[d_idx]
        history_data = history_data[:, :, :, :num_feat]
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = \
            self._prepare_inputs(history_data)

        dataflow_checkpoint("model.raw_input", history_data)
        dump_struct_state(
            "pre_graph",
            history_data=history_data,
            node_emb_u=node_embedding_u,
            node_emb_d=node_embedding_d,
            time_feat=time_in_day_feat)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # 嵌入 + Swish + 可学习缩放
        history_data = self.embedding(history_data)
        history_data = _swish(history_data)
        history_data = (history_data * self.embed_scale
                        + self.embed_bias)
        dataflow_checkpoint("model.post_embed", history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data

        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fh, inh_fh = layer(
                inh_backcast_seq_res, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fh)
            inh_forecast_hidden_list.append(inh_fh)

        # EMA衰减聚合: w_i = decay^(N-1-i), 越后面的层权重越大
        decay = torch.sigmoid(self.ema_decay)
        N = self._num_layers
        raw_weights = torch.stack([
            decay ** (N - 1 - i) for i in range(N)])
        weights = raw_weights / (raw_weights.sum() + 1e-8)

        _dbg("ema_decay", decay, "model")
        _dbg("ema_weights", weights, "model")

        dif_forecast_hidden = sum(
            weights[i] * dif_forecast_hidden_list[i]
            for i in range(N))
        inh_forecast_hidden = sum(
            weights[i] * inh_forecast_hidden_list[i]
            for i in range(N))

        forecast_hidden = (dif_forecast_hidden
                           + inh_forecast_hidden)

        # 输出: 主路 + 残差shortcut
        main_path = self.out_fc_2(
            _swish(self.out_ln(
                _swish(self.out_fc_1(forecast_hidden)))))
        shortcut = self.out_shortcut(forecast_hidden)
        forecast = main_path + 0.1 * shortcut

        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)

        dataflow_checkpoint("model.output", forecast)
        _dbg("output.range",
             f"[{forecast.min().item():.4f},"
             f"{forecast.max().item():.4f}]", "model")
        return forecast
