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
  6. [Phase 3] 空间自注意力 (from STAEformer): 节点维multi-head attention,
     门控残差注入+Pre-LN+轻量FFN+时间分块, 入口空间/出口时序对称结构
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


class TemporalCrossAttention(nn.Module):
    """输出头时序自注意力 — 从STAEformer的alternating T/S attention移植
    在cascade聚合后，对时间维度做single-head self-attention
    捕获输出序列内部的时序依赖（e.g. 第3步和第6步的关联）
    """
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, D] — 对T维度做self-attention
        B, T, N, D = x.shape
        residual = x
        x = self.ln(x)
        # reshape: 合并B和N → [B*N, T, D]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        qkv = self.qkv(x_flat).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*N, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.out_proj(out)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]
        return residual + out


class SpatialSelfAttention(nn.Module):
    """空间自注意力 — 从STAEformer的spatial AttentionLayer移植 (Phase 3 鲁迅拿法)
    对节点维度N做multi-head self-attention: 动态图卷积只能传播k_s跳邻域,
    attention可以一步连接任意两个传感器, 捕获超越图结构的全局空间依赖。
    改写点 (~20%, vs upstream/staeformer/STAEformer.py:AttentionLayer):
      1. 门控残差注入: sigmoid(spa_gate)控制强度, 初始-3.0→0.047极温和启动, 由训练自学开门(与CL sigmoid ramp同哲学)
         (STAEformer是硬残差x+attn, 在已收敛的cascade骨干上硬注入会扰动训练)
      2. Pre-LayerNorm (STAEformer是Post-LN) — 深骨干上pre-norm梯度更稳
      3. 轻量FFN dim*2 (STAEformer用2048) — 控制参数量与显存
      4. 时间维分块计算 — 与本项目chunked gconv同思路, 控制[B*L,h,N,N]峰值
      5. 注意力熵诊断 — 监控attention是否退化(均匀=log N)或坍缩(→0)
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, time_chunk=6):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.time_chunk = time_chunk
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)
        # 门控: sigmoid(-3.0)≈0.047, 早期几乎不扰动骨干, 训练中自学注入强度
        self.spa_gate = nn.Parameter(torch.tensor(-3.0))
        self._last_entropy = None  # 断点诊断: 最近一次注意力熵

    def _attend(self, x_flat):
        # x_flat: [B*Lc, N, D] — attention over node dim N
        BL, N, D = x_flat.shape
        qkv = self.qkv(x_flat).reshape(
            BL, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, BL, h, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [BL, h, N, N]
        attn = attn.softmax(dim=-1)
        with torch.no_grad():
            self._last_entropy = -(
                attn * attn.clamp(min=1e-9).log()
            ).sum(-1).mean()
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(BL, N, D)
        return self.out_proj(out)

    def forward(self, x):
        # x: [B, L, N, D]
        B, L, N, D = x.shape
        h = self.ln(x)  # pre-norm
        outs = []
        for s in range(0, L, self.time_chunk):
            chunk = h[:, s:s + self.time_chunk]
            Lc = chunk.shape[1]
            o = self._attend(chunk.reshape(B * Lc, N, D))
            outs.append(o.reshape(B, Lc, N, D))
        attn_out = torch.cat(outs, dim=1)
        gate = torch.sigmoid(self.spa_gate)
        x = x + gate * attn_out                  # 门控残差注入
        x = x + gate * self.ffn(self.ffn_ln(x))  # 轻量FFN, 同门控
        _dbg("spatial_attn.gate", gate, "model")
        if self._last_entropy is not None:
            _dbg("spatial_attn.entropy", self._last_entropy, "model")
        _dbg("spatial_attn.out_norm",
             attn_out.detach().norm(), "model")
        return x


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
        self._seq_length = model_args['seq_length']

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

        # ═══ 从STAEformer移植: 自适应时空嵌入 ═══
        # learnable [L, N, d_adp] 参数 — STAEformer的核心创新
        # 捕获每个时间步、每个节点的独特模式
        # 原始STAEformer用d_adp=80，我们按比例缩放到hidden_dim的1/4
        self._adp_emb_dim = max(self._hidden_dim // 4, 8)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(
                self._seq_length, self._num_nodes, self._adp_emb_dim)))
        # 投影到hidden_dim并融合
        self.adp_proj = nn.Linear(self._adp_emb_dim, self._hidden_dim, bias=False)
        # 融合门控: 控制adaptive embedding的注入强度
        self.adp_gate = nn.Parameter(torch.tensor(0.3))

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

        # ═══ 从STAEformer移植: 输出头时序自注意力 ═══
        # 在cascade聚合后、regression前插入temporal cross-attention
        # 捕获输出序列步之间的依赖关系
        self.output_temporal_attn = TemporalCrossAttention(
            self._forecast_dim, num_heads=2,
            dropout=model_args.get('dropout', 0.1))

        # ═══ 从STAEformer移植: 空间自注意力 (Phase 3) ═══
        # 在embedding+自适应嵌入后、decouple层前对节点维做self-attention
        # 与输出头temporal attention形成"入口空间/出口时序"对称结构
        # use_spatial_attn: False 时完全跳过 (消融实验用)
        self._use_spatial_attn = model_args.get(
            'use_spatial_attn', True)
        if self._use_spatial_attn:
            self.spatial_attn = SpatialSelfAttention(
                self._hidden_dim, num_heads=4,
                dropout=model_args.get('dropout', 0.1))

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
            # Symmetrize: traffic graphs are largely bidirectional
            raw_adj = 0.5 * (raw_adj + raw_adj.T)
            # Learnable temperature for sharper/softer attention
            adj = F.softmax(F.relu(raw_adj) / self._adj_temperature, dim=1)
            static_graph = [adj]
            _dbg("adj_temperature", self._adj_temperature, "model")
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

        # ═══ 自适应时空嵌入注入 (从STAEformer移植) ═══
        # adaptive_embedding: [L, N, d_adp] → 投影到 [L, N, hidden_dim]
        B = history_data.shape[0]
        L_actual = history_data.shape[1]
        adp_emb = self.adaptive_embedding[:L_actual]  # 截断到实际长度
        adp_feat = self.adp_proj(adp_emb)  # [L, N, hidden_dim]
        adp_feat = adp_feat.unsqueeze(0).expand(B, -1, -1, -1)  # [B, L, N, hidden_dim]
        # 门控注入: 用sigmoid gate控制adaptive embedding的影响
        gate_val = torch.sigmoid(self.adp_gate)
        history_data = history_data + gate_val * adp_feat
        _dbg("adaptive_emb.gate", gate_val, "model")
        _dbg("adaptive_emb.feat_norm", adp_feat.detach().norm(), "model")
        dataflow_checkpoint("model.post_adaptive_emb", history_data)

        # ═══ 空间自注意力 (从STAEformer移植, Phase 3) ═══
        if self._use_spatial_attn:
            history_data = self.spatial_attn(history_data)
            dataflow_checkpoint(
                "model.post_spatial_attn", history_data)
            dump_struct_state(
                "spatial_attn",
                features=history_data,
                gate_raw=float(self.spatial_attn.spa_gate.item()),
                attn_entropy=(
                    float(self.spatial_attn._last_entropy.item())
                    if self.spatial_attn._last_entropy is not None
                    else -1.0))

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

        # ═══ 输出头时序自注意力 (从STAEformer移植) ═══
        # 在regression前，对forecast_hidden的时间维度做self-attention
        # 捕获输出序列步间的依赖
        forecast_hidden = self.output_temporal_attn(forecast_hidden)
        _dbg("output_temporal_attn.out_norm",
             forecast_hidden.detach().norm(), "model")

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
