"""Prism gate: Multi-view attention gating with spatial-frequency awareness.
Unlike upstream (2-layer FC+ReLU+sigmoid) and vortex (SE with spatial attention),
Prism uses a dual-path gate that processes node+time features through both a
spatial attention path and a frequency-aware path, then fuses with learnable blend."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
_PR_DBG = os.environ.get('PRISM_DEBUG', '0') == '1'


class FreqAwarePath(nn.Module):
    """频率感知路径: 对gate特征做频谱分析后提取门控信号"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.freq_fc = nn.Linear(hidden_dim, hidden_dim)
        self.phase_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, L, N, D]
        B, L, N, D = x.shape
        # 沿时间维度做FFT
        x_freq = torch.fft.rfft(x, dim=1)
        # 幅度和相位分别处理
        magnitude = self.freq_fc(x_freq.abs())
        phase = self.phase_fc(x_freq.angle())
        # 重构回时域作为门控信号
        combined = magnitude * torch.exp(
            1j * phase.to(torch.cfloat))
        gate_signal = torch.fft.irfft(combined, n=L, dim=1)
        return torch.sigmoid(gate_signal)


class EstimationGate(nn.Module):
    """Prism estimation gate: Multi-view attention gating.
    Compared to upstream (simple FC+ReLU+sigmoid) and vortex (SE+spatial),
    Prism uses dual-path (spatial + frequency) gating with learnable fusion."""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim,
                 num_nodes=None):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # 空间路径: 标准FC
        self.spatial_fc1 = nn.Linear(in_dim, hidden_dim)
        self.spatial_fc2 = nn.Linear(hidden_dim, 1)
        # 频率感知路径
        self.freq_path = FreqAwarePath(hidden_dim)
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        # 双路融合权重
        self.path_blend = nn.Parameter(torch.tensor(0.7))
        if _PR_DBG:
            print(f"[PR:EstimationGate] in_dim={in_dim} "
                  f"hidden={hidden_dim}", file=sys.stderr)

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat,
                history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        # 空间路径: FC -> ReLU -> FC -> sigmoid
        spatial_hidden = F.relu(
            self.spatial_fc1(estimation_gate_feat))
        spatial_gate = torch.sigmoid(
            self.spatial_fc2(spatial_hidden))
        spatial_gate = spatial_gate[
            :, -history_data.shape[1]:, :, :]
        # 频率路径: 投影到hidden_dim后做频率感知门控
        freq_input = F.relu(
            self.feat_proj(estimation_gate_feat))
        freq_input = freq_input[
            :, -history_data.shape[1]:, :, :]
        freq_gate = self.freq_path(freq_input)
        # mean pool到单通道门控
        freq_gate_scalar = freq_gate.mean(dim=-1,
                                          keepdim=True)
        # 双路融合
        alpha = torch.sigmoid(self.path_blend)
        combined_gate = (alpha * spatial_gate +
                         (1 - alpha) * freq_gate_scalar)
        if _PR_DBG:
            print(f"[PR:EstimationGate] alpha={alpha.item():.4f} "
                  f"spatial_mean={spatial_gate.mean().item():.4f} "
                  f"freq_mean={freq_gate_scalar.mean().item():.4f}",
                  file=sys.stderr)
        history_data = history_data * combined_gate
        return history_data
