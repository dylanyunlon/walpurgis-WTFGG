"""
D2STGNN — Helix变体
算法改写 (~20%):
  1. DecoupleLayer: 螺旋门控 — dif/inh分支通过helix phase rotation混合,
     交替升维-降维螺旋结构替代直接pass-through
  2. D2STGNN: top-k自适应图稀疏化 — 对static graph做可学习的top-k mask
  3. D2STGNN: 螺旋位置编码注入 — 用旋转相位调制embedding
  4. 输出头: GELU激活 + LayerNorm + helix channel rotation
  5. embedding后接螺旋通道交错 (alternating expand-contract)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


class HelixConv(nn.Module):
    """Helix特有: 螺旋卷积 — 交替升维-降维螺旋结构
    expand_ratio控制膨胀倍数, 先升维到hidden*expand再降回hidden,
    中间用旋转相位gate控制信息流"""
    def __init__(self, hidden_dim, expand_ratio=2):
        super().__init__()
        mid_dim = hidden_dim * expand_ratio
        self.expand = nn.Linear(hidden_dim, mid_dim)
        self.contract = nn.Linear(mid_dim, hidden_dim)
        # 螺旋相位参数: 可学习的旋转角度
        self.phase = nn.Parameter(torch.zeros(mid_dim))
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        h = self.expand(x)
        # 螺旋gate: cos(phase)做幅度调制, sin(phase)做相位偏移
        cos_gate = torch.cos(self.phase)
        sin_gate = torch.sin(self.phase)
        h = h * cos_gate + torch.roll(h, 1, dims=-1) * sin_gate
        h = F.gelu(h)
        h = self.contract(h)
        return self.ln(h + residual)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256,
                 layer_idx=0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
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
        # Helix特有: 螺旋卷积在dif/inh分支之间做交替变换
        self.helix_conv = HelixConv(hidden_dim, expand_ratio=2)
        # Helix特有: 层间螺旋相位 — 可学习的旋转混合权重
        self.helix_alpha = nn.Parameter(torch.tensor(0.5))
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        dataflow_checkpoint(
            f"decouple_L{self.layer_idx}.input",
            history_data)
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
        # Helix特有: 对inh backcast施加螺旋卷积变换
        inh_backcast_seq_res = self.helix_conv(
            inh_backcast_seq_res)
        # Helix特有: 用可学习alpha在原始和螺旋变换之间插值
        alpha = torch.sigmoid(self.helix_alpha)
        inh_backcast_seq_res = (
            alpha * inh_backcast_seq_res +
            (1 - alpha) * dif_backcast_seq_res[
                :, -inh_backcast_seq_res.shape[1]:, :, :])
        _dbg(f"decouple_L{self.layer_idx}.helix_alpha",
             alpha, "model")
        _dbg(f"decouple_L{self.layer_idx}.dif_energy",
             dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy",
             inh_forecast_hidden.detach().norm(), "model")
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
        # embedding + 螺旋通道交错 (Helix特有)
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        # Helix特有: 螺旋位置编码注入 — 用sin/cos对通道维度做旋转
        self.helix_phase_embed = nn.Parameter(
            torch.randn(self._hidden_dim) * 0.01)
        self.helix_amplitude = nn.Parameter(
            torch.ones(self._hidden_dim) * 0.1)
        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))
        # decouple layers
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])
        # Helix特有: top-k自适应图稀疏化
        # 可学习的k比例参数，控制保留多少边
        self._topk_ratio = nn.Parameter(torch.tensor(0.5))
        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)
        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        # 输出头: GELU + LayerNorm + helix channel rotation (Helix特有)
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # Helix特有: 输出螺旋旋转矩阵 (轻量)
        self.out_helix_rotate = nn.Parameter(
            torch.eye(model_args['gap']) * 0.9 +
            torch.randn(model_args['gap'],
                        model_args['gap']) * 0.1)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _topk_sparsify(self, graph):
        """Helix特有: top-k自适应图稀疏化
        保留每行top-k个最大值, 其余置零
        k由可学习参数_topk_ratio控制"""
        ratio = torch.sigmoid(self._topk_ratio)
        N = graph.shape[-1]
        k = max(1, int(ratio.item() * N))
        _dbg("topk_sparsify.ratio", ratio, "model")
        _dbg("topk_sparsify.k", f"{k}/{N}", "model")
        # 保留top-k
        topk_vals, topk_idx = torch.topk(graph, k, dim=-1)
        sparse_graph = torch.zeros_like(graph)
        sparse_graph.scatter_(-1, topk_idx, topk_vals)
        # 重新归一化
        row_sum = sparse_graph.sum(dim=-1, keepdim=True)
        row_sum = row_sum.clamp(min=1e-8)
        sparse_graph = sparse_graph / row_sum
        return sparse_graph

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            raw_graph = F.softmax(
                F.relu(torch.mm(E_d, E_u.T)), dim=1)
            # Helix特有: 对static graph做top-k稀疏化
            static_graph = [self._topk_sparsify(raw_graph)]
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
            time_feat_shape=time_in_day_feat)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)
        # embedding + 螺旋位置编码注入 (Helix特有)
        history_data = self.embedding(history_data)
        # 螺旋调制: amplitude * sin(phase + position_index)
        B, L, N, D = history_data.shape
        pos_idx = torch.arange(L, device=history_data.device
                               ).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        helix_modulation = self.helix_amplitude * torch.sin(
            self.helix_phase_embed + pos_idx * 0.1)
        history_data = history_data + helix_modulation
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
        # 聚合: sum (与upstream一致)
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = (dif_forecast_hidden +
                           inh_forecast_hidden)
        # 输出头: GELU + LayerNorm + helix rotation (Helix特有)
        h = F.gelu(self.out_fc_1(forecast_hidden))
        # LayerNorm替代ReLU
        h = self.out_ln(h)
        forecast = self.out_fc_2(F.gelu(h))
        # Helix特有: 输出螺旋旋转 — 对gap维度做可学习的线性变换
        forecast = torch.matmul(
            forecast, self.out_helix_rotate)
        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.range",
             f"[{forecast.min().item():.4f},"
             f"{forecast.max().item():.4f}]", "model")
        _dbg("output.topk_ratio",
             torch.sigmoid(self._topk_ratio), "model")
        return forecast
