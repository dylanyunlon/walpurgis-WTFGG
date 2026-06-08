"""
D2STGNN — Zenith变体
算法改写 (~20%):
  1. DecoupleLayer: 频域衰减门控 (spectral decay gate)
     用可学习的频率权重对history_data做傅里叶域过滤再送入estimation_gate
  2. D2STGNN: layer-wise attention聚合 (替代sum)
     每层输出经全局平均池化后过attention打分, 加权聚合
  3. 输出头: GELU + LayerNorm + 残差shortcut
  4. 每层可学习residual_scale, 控制跨层残差传播幅度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint


class SpectralDecayGate(nn.Module):
    """频域衰减门控: 在频率域对序列做可学习的衰减加权"""
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.freq_weights = nn.Parameter(torch.ones(n_freq, hidden_dim))
        nn.init.normal_(self.freq_weights, mean=1.0, std=0.05)

    def forward(self, x):
        # x: [B, L, N, D]
        x_freq = torch.fft.rfft(x, dim=1)
        n_freq = x_freq.shape[1]
        w = torch.sigmoid(self.freq_weights[:n_freq, :])
        x_filtered = x_freq * w.unsqueeze(0).unsqueeze(2)
        out = torch.fft.irfft(x_filtered, n=x.shape[1], dim=1)
        return out


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
        # 可学习残差缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

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
        # 可学习残差缩放
        scale = torch.sigmoid(self.residual_scale)
        inh_backcast_seq_res = inh_backcast_seq_res * scale
        _dbg(f"decouple_L{self.layer_idx}.res_scale",
             scale, "model")
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class LayerAttentionAggregator(nn.Module):
    """Layer-wise attention聚合: 对各层hidden做注意力加权"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_layers = num_layers

    def forward(self, hidden_list):
        # hidden_list: list of [B, L, N, D]
        # 全局平均池化 -> [num_layers, D]
        pooled = []
        for h in hidden_list:
            pooled.append(h.mean(dim=(0, 1, 2)))  # [D]
        pooled = torch.stack(pooled, dim=0)  # [num_layers, D]
        q = self.query(pooled)               # [num_layers, D]
        k = self.key(pooled)                 # [num_layers, D]
        attn = torch.matmul(q, k.T) / (q.shape[-1] ** 0.5)  # [L, L]
        attn_weights = F.softmax(attn.mean(dim=1), dim=0)    # [L]
        _dbg("layer_attn_weights", attn_weights, "model")
        result = sum(
            attn_weights[i] * hidden_list[i]
            for i in range(self.num_layers))
        return result


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
        self.emb_ln = nn.LayerNorm(self._hidden_dim)
        # 频域衰减门控 (Zenith特有)
        self.spectral_gate = SpectralDecayGate(
            seq_len=model_args['seq_length'],
            hidden_dim=self._hidden_dim)
        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))
        # decouple layers, 每层带layer_idx
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])
        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)
        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        # layer-wise attention聚合 (Zenith特有)
        self.dif_aggregator = LayerAttentionAggregator(
            self._forecast_dim, self._num_layers)
        self.inh_aggregator = LayerAttentionAggregator(
            self._forecast_dim, self._num_layers)
        # 输出头: GELU + LayerNorm + 残差
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
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
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = \
            self._prepare_inputs(history_data)
        dataflow_checkpoint("model.raw_input", history_data)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)
        # embedding + LayerNorm
        history_data = self.embedding(history_data)
        history_data = self.emb_ln(history_data)
        # 频域衰减门控
        history_data = self.spectral_gate(history_data)
        dataflow_checkpoint("model.post_spectral", history_data)
        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fh, inh_fh = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fh)
            inh_forecast_hidden_list.append(inh_fh)
        # layer-wise attention聚合
        dif_forecast_hidden = self.dif_aggregator(dif_forecast_hidden_list)
        inh_forecast_hidden = self.inh_aggregator(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden
        # GELU输出头 + 残差shortcut
        h = F.gelu(self.out_fc_1(F.relu(forecast_hidden)))
        h = self.out_ln(h)
        main_out = self.out_fc_2(h)
        shortcut = self.out_shortcut(forecast_hidden)
        forecast = main_out + 0.1 * shortcut
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        dataflow_checkpoint("model.output", forecast)
        return forecast
