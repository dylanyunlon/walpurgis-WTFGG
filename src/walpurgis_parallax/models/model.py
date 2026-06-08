"""
D2STGNN — Parallax变体 (M054)
算法改写 (~20%):
  1. Mixture Output Router: 各层输出通过路由网络学习权重
     不再用EMA衰减, 而是gating network动态选择每层贡献
     路由网络输入: 各层forecast hidden的统计量(均值+方差)
     输出: softmax权重, 每层一个标量
  2. 嵌入后接GELU + 可学习偏差缩放
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


class MixtureRouter(nn.Module):
    """Mixture Output Router — 学习每层的输出权重

    输入: 各层forecast hidden列表
    输出: softmax加权的聚合结果

    与Penumbra的EMA衰减对比:
      EMA: 固定衰减模式, 越后面的层权重越大
      Router: 根据每层输出的统计量动态决定权重
             不同样本可能有不同的最优层组合
    """

    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        # 路由网络: 每层输入2维统计量(均值+标准差)
        # → 映射到该层的权重
        self.route_fc = nn.Sequential(
            nn.Linear(num_layers * 2, num_layers * 4),
            nn.GELU(),
            nn.Linear(num_layers * 4, num_layers),
        )
        self.num_layers = num_layers
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_list):
        """
        hidden_list: list of [B, T, N, D] tensors
        returns: weighted sum [B, T, N, D]
        """
        # 收集每层的统计量
        stats = []
        for h in hidden_list:
            h_flat = h.reshape(h.shape[0], -1)
            stats.append(h_flat.mean(dim=-1, keepdim=True))
            stats.append(h_flat.std(dim=-1, keepdim=True))
        # [B, num_layers*2]
        stats = torch.cat(stats, dim=-1)

        # 路由权重: softmax with温度
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        logits = self.route_fc(stats) / temp
        weights = F.softmax(logits, dim=-1)  # [B, num_layers]

        _dbg("router.weights_mean",
             weights.mean(dim=0), "model")
        _dbg("router.temperature", temp, "model")

        # 加权聚合
        result = torch.zeros_like(hidden_list[0])
        for i, h in enumerate(hidden_list):
            w = weights[:, i].reshape(-1, 1, 1, 1)
            result = result + w * h
        return result


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

        # 嵌入层 + GELU + 可学习缩放
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

        # Mixture Output Router替代EMA衰减聚合
        self.dif_router = MixtureRouter(
            self._num_layers, self._forecast_dim)
        self.inh_router = MixtureRouter(
            self._num_layers, self._forecast_dim)

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
        # 残差shortcut
        self.out_shortcut = nn.Linear(
            self._forecast_dim, model_args['gap'])

        self.reset_parameter()

        print(f"[PAR-DBG] D2STGNN Parallax: "
              f"layers={self._num_layers}, "
              f"hidden={self._hidden_dim}, "
              f"forecast={self._forecast_dim}, "
              f"router=MixtureRouter")

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

        # 嵌入 + GELU + 可学习缩放
        history_data = self.embedding(history_data)
        history_data = F.gelu(history_data)
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

        # Mixture Output Router: 动态路由各层输出
        dif_forecast_hidden = self.dif_router(
            dif_forecast_hidden_list)
        inh_forecast_hidden = self.inh_router(
            inh_forecast_hidden_list)

        forecast_hidden = (dif_forecast_hidden
                           + inh_forecast_hidden)

        # 输出: 主路 + 残差shortcut
        main_path = self.out_fc_2(
            F.gelu(self.out_ln(
                F.gelu(self.out_fc_1(forecast_hidden)))))
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
