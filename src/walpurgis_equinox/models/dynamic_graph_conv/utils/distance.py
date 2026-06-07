import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:dist:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DistanceFunction(nn.Module):
    """upstream: scaled-dot attention O(N^2)
    equinox: Linformer低秩投影注意力 + Gumbel-Softmax离散图采样"""
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']

        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)

        # equinox: WeightNorm替代BN
        self.wn_proj = nn.utils.weight_norm(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2))

        # equinox: Linformer投影矩阵 (N -> proj_dim)
        num_nodes = model_args['num_nodes']
        self.linf_proj_dim = min(8, num_nodes)
        self.linf_E = nn.Parameter(torch.randn(self.linf_proj_dim, num_nodes) * 0.02)
        self.linf_F = nn.Parameter(torch.randn(self.linf_proj_dim, num_nodes) * 0.02)

        # equinox: Gumbel-Softmax温度 (可退火)
        self.gumbel_tau = nn.Parameter(torch.tensor(1.0))

    def _gumbel_softmax_sample(self, logits, hard=False):
        """equinox: Gumbel-Softmax离散图采样
        训练时用可微的soft近似, 推理时可切换到hard one-hot"""
        tau = torch.clamp(self.gumbel_tau, min=0.1, max=5.0)
        return F.gumbel_softmax(logits, tau=tau.item(), hard=hard, dim=-1)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)

        # equinox: WeightNorm + Mish
        h = self.fc_ts_emb1(X)
        h = h.view(B, N, -1)
        h = self.wn_proj(h)
        h = F.mish(h)
        h = h.view(B * N, -1)
        dy_feat = self.fc_ts_emb2(self.dropout(h))
        dy_feat = dy_feat.view(B, N, -1)

        emb1 = E_d.unsqueeze(0).expand(B, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(B, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        adjacent_list = []
        for feat in [X1, X2]:
            Q = self.WQ(feat)   # [B, N, D]
            K = self.WK(feat)   # [B, N, D]

            # equinox: Linformer低秩投影 K: [B, N, D] -> [B, proj, D]
            K_proj = torch.matmul(self.linf_E[:, :N].unsqueeze(0).expand(B, -1, -1).to(Q.device), K)
            # Attention: [B, N, D] @ [B, D, proj] -> [B, N, proj]
            attn_logits = torch.bmm(Q, K_proj.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

            # equinox: Gumbel-Softmax离散图采样
            # 将logits通过Gumbel-Softmax采样, 生成近似离散的稀疏邻接矩阵
            W_gumbel = self._gumbel_softmax_sample(attn_logits, hard=not self.training)

            # 还原到 [B, N, N]: 通过与F投影相乘扩展
            F_proj = self.linf_F[:, :N].unsqueeze(0).expand(B, -1, -1).to(Q.device)
            W = torch.bmm(W_gumbel, F_proj)  # [B, N, N]
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)

            _edbg("gumbel_attn", W)
            _edbg("gumbel_tau", self.gumbel_tau)
            adjacent_list.append(W)
        return adjacent_list
