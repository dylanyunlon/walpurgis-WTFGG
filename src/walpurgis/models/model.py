"""
D2STGNN — Cascade变体
算法改写 (~20%):
  1. DecoupleLayer: SE通道注意力 — 在dif/inh融合前对通道做
     squeeze-and-excitation自适应加权
  2. D2STGNN: 级联残差学习 — 每层的backcast输出直接跳连到最终聚合
     而非仅传递给下一层, 形成dense cascade连接
  3. D2STGNN: 动态深度门控 — 可学习sigmoid门控决定每层的
     forecast贡献是否被采纳, 实现推理时的动态深度
  4. 输出头: GELU激活 + LayerNorm + 残差shortcut
  5. embedding后接SE通道注意力块 (Cascade特有)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


class SqueezeExcitation(nn.Module):
    """Cascade特有: SE通道注意力块
    对hidden_dim维度做squeeze(全局平均池化)->excitation(FC降维->ReLU->FC升维->Sigmoid)
    输出通道权重在[0,1], 用于自适应加权各通道的重要性
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(channels)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, N, C]
        B, L, N, C = x.shape
        # LayerNorm stabilizes channel statistics before squeeze
        x_normed = self.ln(x)
        # squeeze: 对空间维度N做全局平均
        squeezed = x_normed.mean(dim=2)  # [B, L, C]
        squeezed = squeezed.mean(dim=1)  # [B, C]
        # excitation
        weights = self.excitation(squeezed)  # [B, C]
        # broadcast回原始shape
        weights = weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
        return x * weights, weights.squeeze()


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256,
                 layer_idx=0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=hidden_dim)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)
        # Cascade特有: SE通道注意力应用于融合后的backcast
        self.se_block = SqueezeExcitation(hidden_dim, reduction=4)
        # 每层到cascade输出的投影 (用于cascade residual)
        self.cascade_proj = nn.Linear(hidden_dim, fk_dim)
        # 频率域残差门控: 从cascade特征中提取周期性成分注回
        self.freq_gate = nn.Parameter(torch.tensor(0.1))
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un decoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: cascade residual — 该层backcast的投影, 跳连到最终输出聚合
        """
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
        # Cascade特有: SE通道注意力对backcast加权
        inh_backcast_seq_res, se_weights = self.se_block(
            inh_backcast_seq_res)
        # Cascade特有: 生成cascade residual (该层对输出的直接贡献)
        # 用时间维度均值池化(比单取最后一步更稳定,梯度信号更丰富)
        cascade_feat = inh_backcast_seq_res.mean(dim=1, keepdim=True)  # [B, 1, N, D]
        fk_len = self._seq_length // self._gap
        cascade_residual = self.cascade_proj(cascade_feat)  # [B, 1, N, fk_dim]
        # 频率域残差注入: 提取cascade_feat周期成分，增强长程pattern捕获
        freq_input = cascade_feat.squeeze(1)  # [B, N, D]
        if freq_input.shape[-1] > 1:
            cascade_fft = torch.fft.rfft(freq_input, dim=-1)
            freq_mag = cascade_fft.abs().mean()
            freq_recon = torch.fft.irfft(cascade_fft, n=freq_input.shape[-1], dim=-1)
            freq_proj = self.cascade_proj(freq_recon).unsqueeze(1)  # [B, 1, N, fk_dim]
            cascade_residual = cascade_residual + self.freq_gate * freq_proj
            _dbg(f"decouple_L{self.layer_idx}.freq_mag", freq_mag, "model")
        cascade_residual = cascade_residual.expand(-1, fk_len, -1, -1)  # [B, fk_len, N, fk_dim]
        _dbg(f"decouple_L{self.layer_idx}.se_weights",
             se_weights, "model")
        _dbg(f"decouple_L{self.layer_idx}.dif_energy",
             dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy",
             inh_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.cascade_norm",
             cascade_residual.detach().norm(), "model")
        return (inh_backcast_seq_res,
                dif_forecast_hidden,
                inh_forecast_hidden,
                cascade_residual)


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
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        # Cascade特有: embedding后的SE通道注意力
        self.embed_se = SqueezeExcitation(
            self._hidden_dim, reduction=4)

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])

        # Cascade特有: 动态深度门控 — 每层一个可学习gate
        # sigmoid(3.0) ≈ 0.95, 初始近乎全开, 训练中学习是否关闭某些层
        self.depth_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(3.0))
            for _ in range(self._num_layers)
        ])

        # Cascade特有: 级联残差聚合的可学习权重
        self.cascade_weights = nn.Parameter(
            torch.ones(self._num_layers) / self._num_layers)

        # dynamic and static hidden graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # Adaptive adjacency learning: learnable temperature for static graph
        self._adj_temperature = nn.Parameter(torch.tensor(1.0))

        # 输出头: GELU + LayerNorm + 残差shortcut (Cascade特有)
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # 残差shortcut: forecast_dim -> gap 直连
        self.out_shortcut = nn.Linear(
            self._forecast_dim, model_args['gap'])

        # Feature Refinement Module: gated 2-layer refinement after cascade aggregation
        # Reduces residual noise from multi-layer fusion via learned gating
        self.refine_fc = nn.Linear(self._forecast_dim, self._forecast_dim)
        self.refine_gate = nn.Linear(self._forecast_dim, self._forecast_dim)
        self.refine_ln = nn.LayerNorm(self._forecast_dim)

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
            # Adaptive adjacency learning: learnable graph + temperature-scaled softmax
            # Base graph from node embeddings
            raw_adj = torch.mm(E_d, E_u.T)
            # Learnable temperature for sharper/softer attention
            adj = F.softmax(F.relu(raw_adj) / self._adj_temperature, dim=1)
            static_graph = [adj]
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
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        t_idx = (history_data[:, :, :, num_feat] * 288
                 ).type(torch.LongTensor).clamp(0, 287)
        d_idx = (history_data[:, :, :, num_feat + 1]
                 ).type(torch.LongTensor).clamp(0, 6)
        time_in_day_feat = self.T_i_D_emb[t_idx]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[d_idx]          # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        """Feed forward of D2STGNN (Cascade variant).

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]
        """

        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)
        dataflow_checkpoint("model.raw_input", history_data)
        dump_struct_state(
            "pre_graph",
            history_data=history_data,
            node_emb_u=node_embedding_u,
            node_emb_d=node_embedding_d,
            time_feat_shape=time_in_day_feat)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, history_data=history_data, time_in_day_feat=time_in_day_feat, day_in_week_feat=day_in_week_feat)

        # Start embedding layer + SE通道注意力
        history_data = self.embedding(history_data)
        history_data, _ = self.embed_se(history_data)
        dataflow_checkpoint("model.post_embed_se", history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        cascade_residual_list = []

        inh_backcast_seq_res = history_data
        for layer_idx, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden, cascade_res = layer(inh_backcast_seq_res, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            # Cascade特有: 动态深度门控
            gate = torch.sigmoid(self.depth_gates[layer_idx])
            _dbg(f"depth_gate_L{layer_idx}", gate, "model")
            dif_forecast_hidden_list.append(
                gate * dif_forecast_hidden)
            inh_forecast_hidden_list.append(
                gate * inh_forecast_hidden)
            cascade_residual_list.append(cascade_res)

        # Output Layer: 动态深度门控已应用于各层
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        # Cascade特有: 级联残差聚合 — 将每层的cascade residual加权聚合到forecast
        cascade_w = F.softmax(self.cascade_weights, dim=0)
        _dbg("cascade_weights", cascade_w, "model")
        cascade_aggregate = sum(
            cascade_w[i] * cascade_residual_list[i]
            for i in range(self._num_layers))
        forecast_hidden = forecast_hidden + cascade_aggregate

        # Feature Refinement Module: gated refinement reduces cascade noise
        refine_h = F.gelu(self.refine_fc(forecast_hidden))
        refine_g = torch.sigmoid(self.refine_gate(forecast_hidden))
        forecast_hidden = self.refine_ln(forecast_hidden + refine_g * refine_h)
        _dbg("refine_gate_mean", refine_g.mean(), "model")

        # regression layer: GELU + LayerNorm + 残差shortcut
        h = F.gelu(self.out_fc_1(forecast_hidden))
        # LayerNorm在最后一维
        h = self.out_ln(h)
        main_out = self.out_fc_2(F.gelu(h))
        # 残差shortcut
        shortcut = self.out_shortcut(forecast_hidden)
        forecast = main_out + 0.1 * shortcut

        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.range",
             f"[{main_out.min().item():.4f},"
             f"{main_out.max().item():.4f}]", "model")
        _dbg("output.shortcut_scale",
             shortcut.detach().norm(), "model")

        return forecast
