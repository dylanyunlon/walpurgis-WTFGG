"""
D2STGNN — Rift变体
算法改写 (~20%):
  1. DecoupleLayer: Split-Recombine注意力 — hidden分K组独立处理后
     通过交叉拼接重组, 增强组间信息交流
  2. D2STGNN: FFT域特征增强 — embedding后提取频域特征与时域concat
  3. D2STGNN: 频域残差旁路 — forecast hidden经FFT提取后加回
  4. 输出头: SiLU激活 + RMSNorm + 频谱门控
  5. embedding后接可学习的通道混洗 (channel shuffle via 1x1 conv)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Rift特有)"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class SplitRecombineBlock(nn.Module):
    """Split-Recombine: 将hidden分成K组, 各组独立做线性变换后交叉重组 (Rift特有)"""
    def __init__(self, hidden_dim, num_groups=4, dropout=0.1):
        super().__init__()
        self.num_groups = num_groups
        assert hidden_dim % num_groups == 0
        self.group_dim = hidden_dim // num_groups
        self.group_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.group_dim, self.group_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.group_dim * 2, self.group_dim),
            ) for _ in range(num_groups)
        ])
        self.recombine_proj = nn.Linear(hidden_dim, hidden_dim)
        self.recombine_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L, N, D = x.shape
        groups = x.reshape(B, L, N, self.num_groups, self.group_dim)
        processed = []
        for g in range(self.num_groups):
            group_input = groups[:, :, :, g, :]
            group_out = self.group_transforms[g](group_input)
            processed.append(group_out)
            _dbg(f"split_group_{g}.energy", group_out.detach().norm(), "model")
        stacked = torch.stack(processed, dim=-2)
        interleaved = stacked.permute(0, 1, 2, 4, 3)
        recombined = interleaved.reshape(B, L, N, D)
        gate = torch.sigmoid(self.recombine_gate)
        out = x + gate * self.recombine_proj(recombined)
        _dbg("split_recombine.gate", gate, "model")
        return out


class FFTFeatureExtractor(nn.Module):
    """FFT域特征提取: 时域特征做FFT, 取top-K频率重建, concat回时域 (Rift特有)"""
    def __init__(self, hidden_dim, top_k_freq=4):
        super().__init__()
        self.top_k = top_k_freq
        self.freq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        B, L, N, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        amplitude = x_freq.abs()
        freq_energy = amplitude.sum(dim=(0, 2, 3))
        k = min(self.top_k, freq_energy.shape[0])
        top_indices = torch.topk(freq_energy, k).indices
        mask = torch.zeros_like(x_freq)
        mask[:, top_indices, :, :] = 1.0
        x_filtered = torch.fft.irfft(x_freq * mask, n=L, dim=1)
        freq_feat = self.freq_proj(x_filtered)
        gate = torch.sigmoid(self.fusion_gate)
        combined = torch.cat([x, freq_feat], dim=-1)
        fused = self.fusion(combined)
        out = (1 - gate) * x + gate * fused
        _dbg("fft_extractor.gate", gate, "model")
        return out


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, layer_idx=0, num_groups=4, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.split_recombine = SplitRecombineBlock(
            hidden_dim, num_groups=num_groups, dropout=model_args.get('dropout', 0.1))
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        dataflow_checkpoint(f"decouple_L{self.layer_idx}.input", history_data)
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data, gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        inh_backcast_seq_res = self.split_recombine(inh_backcast_seq_res)
        _dbg(f"decouple_L{self.layer_idx}.dif_energy", dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy", inh_forecast_hidden.detach().norm(), "model")
        return (inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden)


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
        self.channel_shuffle = nn.Conv1d(
            self._hidden_dim, self._hidden_dim, kernel_size=1, groups=min(4, self._hidden_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))
        self.fft_enhancer = FFTFeatureExtractor(self._hidden_dim, top_k_freq=4)
        num_groups = min(4, self._hidden_dim)
        while self._hidden_dim % num_groups != 0:
            num_groups -= 1
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                         layer_idx=i, num_groups=num_groups, **model_args)
            for i in range(self._num_layers)])
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_rms_norm = RMSNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
        self.spectral_bypass = nn.Linear(self._forecast_dim, model_args['gap'])
        self.spectral_gate = nn.Parameter(torch.tensor(0.05))
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
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
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
        t_idx = (history_data[:, :, :, num_feat] * 288).type(torch.LongTensor).clamp(0, 287)
        d_idx = (history_data[:, :, :, num_feat + 1]).type(torch.LongTensor).clamp(0, 6)
        time_in_day_feat = self.T_i_D_emb[t_idx]
        day_in_week_feat = self.D_i_W_emb[d_idx]
        history_data = history_data[:, :, :, :num_feat]
        return (history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)
        dataflow_checkpoint("model.raw_input", history_data)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d,
            history_data=history_data, time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)
        history_data = self.embedding(history_data)
        B, L, N, D = history_data.shape
        h_flat = history_data.reshape(B * L * N, D, 1)
        h_flat = self.channel_shuffle(h_flat)
        history_data = h_flat.reshape(B, L, N, D)
        history_data = self.fft_enhancer(history_data)
        dataflow_checkpoint("model.post_fft_enhance", history_data)
        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fh, inh_fh = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fh)
            inh_forecast_hidden_list.append(inh_fh)
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden
        h = F.silu(self.out_fc_1(forecast_hidden))
        h = self.out_rms_norm(h)
        main_out = self.out_fc_2(F.silu(h))
        fh_freq = torch.fft.rfft(forecast_hidden, dim=1)
        n_keep = min(2, fh_freq.shape[1])
        fh_low = torch.zeros_like(fh_freq)
        fh_low[:, :n_keep, :, :] = fh_freq[:, :n_keep, :, :]
        fh_low_time = torch.fft.irfft(fh_low, n=forecast_hidden.shape[1], dim=1)
        spectral_out = self.spectral_bypass(fh_low_time)
        gate = torch.sigmoid(self.spectral_gate)
        forecast = main_out + gate * spectral_out
        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(forecast.shape[0], forecast.shape[1], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.spectral_gate", gate, "model")
        return forecast
