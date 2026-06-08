"""
D2STGNN — Aurora变体
算法改写 (~20%):
  1. DecoupleLayer: Gated Fusion — 用sigmoid门控替代简单残差加法
     融合diffusion和inherent分支的输出, 让网络自适应学习两个分支的混合比例
  2. D2STGNN: 将DynamicGraphConstructor的spectral_reg_loss传递给trainer
  3. 输出层使用SiLU替代第一个ReLU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, layer_idx=0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

        # Aurora算法改动2: Gated Fusion
        # 用sigmoid门控替代简单的串行残差传递
        # dif_backcast和inh_backcast通过学习的gate加权融合
        self.fusion_gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        dataflow_checkpoint(
            f"decouple_L{self.layer_idx}.input", history_data)

        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)

        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data,
            gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph,
            static_graph=static_graph)

        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(
            dif_backcast_seq_res)

        # Aurora: Gated Fusion — 用sigmoid gate融合两个分支
        # 而非原始的简单串行(dif→inh)
        # gate ∈ (0,1) 控制保留多少inherent vs diffusion信息
        gate_input = torch.cat([
            dif_backcast_seq_res[:, -inh_backcast_seq_res.shape[1]:, :, :],
            inh_backcast_seq_res
        ], dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate_linear(gate_input))
        fused_output = fusion_gate * inh_backcast_seq_res + \
            (1 - fusion_gate) * dif_backcast_seq_res[:, -inh_backcast_seq_res.shape[1]:, :, :]
        fused_output = self.fusion_proj(fused_output)

        _dbg(f"decouple_L{self.layer_idx}.fusion_gate_mean",
             fusion_gate.mean(), "model")

        return fused_output, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
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

        # start embedding layer
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layers with layer index
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])

        # dynamic and static graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # Aurora: 输出层用SiLU替代第一个ReLU
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # 存储graph正则化损失
        self.graph_reg_loss = torch.tensor(0.0)

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
            # Aurora: 提取spectral正则化损失
            self.graph_reg_loss = self.dynamic_graph_constructor.graph_reg_loss
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
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        # Prepare inputs
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = \
            self._prepare_inputs(history_data)

        dataflow_checkpoint("model.raw_input", history_data)

        # Construct graphs
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # Start embedding
        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        # Aurora: SiLU替代第一个ReLU, 保持第二个ReLU
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.silu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        dataflow_checkpoint("model.output", forecast)
        return forecast
