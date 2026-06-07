import torch
import torch.nn as nn
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:gate:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:gate:{tag}] {val}", file=sys.stderr)


class LinformerProjection(nn.Module):
    """equinox: Linformer低秩投影层
    将K,V从seq_len维投影到proj_dim维, 将O(N^2)注意力降为O(N*k)
    适合大规模节点图的计算效率优化
    操作: E @ K, 其中 K: [..., seq_len, dim] -> [..., proj_dim, dim]"""
    def __init__(self, seq_len, proj_dim=8):
        super().__init__()
        self.E = nn.Parameter(torch.randn(proj_dim, seq_len) * 0.02)
        self.F_proj = nn.Parameter(torch.randn(proj_dim, seq_len) * 0.02)

    def forward(self, K, V):
        # K, V: [..., seq_len, dim]
        # Project seq_len -> proj_dim
        K_proj = torch.matmul(self.E.to(K.device), K)   # [..., proj_dim, dim]
        V_proj = torch.matmul(self.F_proj.to(V.device), V)
        return K_proj, V_proj


class EstimationGate(nn.Module):
    """upstream: 2层FC+ReLU门控
    equinox: 3层FC + Mish激活 + Linformer低秩注意力门 + 可学习温度τ"""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # equinox: 3层FC
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # equinox: Mish激活
        self.act = nn.Mish()
        # equinox: Linformer低秩投影注意力
        # 对节点维度N做投影: N -> proj_dim
        self.linformer_proj_dim = 8
        # 延迟初始化E/F, 因为N在运行时才知道
        self._linformer_init = False
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = hidden_dim ** -0.5
        # equinox: 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _ensure_linformer(self, N, device):
        if not self._linformer_init or self._linformer_N != N:
            proj_dim = min(self.linformer_proj_dim, N)
            self._E = nn.Parameter(torch.randn(proj_dim, N, device=device) * 0.02)
            self._F_proj = nn.Parameter(torch.randn(proj_dim, N, device=device) * 0.02)
            self._linformer_N = N
            self._linformer_init = True

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        N = time_in_day_feat.shape[2]
        feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ], dim=-1)

        h = self.act(self.fc1(feat))  # [B, L, N, hidden_dim]

        # equinox: Linformer低秩注意力
        # h: [B, L, N, D] — 对N(节点)维度做低秩投影
        self._ensure_linformer(N, h.device)
        Q = self.attn_q(h)  # [B, L, N, D]
        # 转置使节点维度在倒数第二维: [B, L, N, D]
        # E: [k, N] @ h: [B*L, N, D] -> [B*L, k, D]
        BL = B * L
        h_flat = h.reshape(BL, N, -1)  # [BL, N, D]
        K_proj = torch.matmul(self._E.to(h.device), h_flat)  # [BL, k, D]
        V_proj = torch.matmul(self._F_proj.to(h.device), h_flat)  # [BL, k, D]
        Q_flat = Q.reshape(BL, N, -1)  # [BL, N, D]
        # Attention: [BL, N, D] @ [BL, D, k] -> [BL, N, k]
        attn_scores = torch.matmul(Q_flat, K_proj.transpose(-1, -2)) * self.attn_scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [BL, N, k]
        # [BL, N, k] @ [BL, k, D] -> [BL, N, D]
        h_attn = torch.matmul(attn_weights, V_proj)
        h = h_attn.reshape(B, L, N, -1)
        _edbg("linformer_attn", attn_weights.mean())

        h = self.act(self.fc2(h))
        # equinox: 温度缩放sigmoid
        tau = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        _edbg("gate_val", gate)
        _edbg("temperature", tau)
        return history_data * gate
