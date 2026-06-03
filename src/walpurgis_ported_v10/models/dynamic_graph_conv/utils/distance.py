import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

_TAG = "dist"

# 改动1: 多头注意力 head 数
_NUM_HEADS = 3


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.input_seq_len = model_args['seq_length']

        # 时序特征提取
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # 改动2: BN → InstanceNorm1d
        # InstanceNorm 对每个 sample 独立, 不依赖 batch 统计量
        self.norm = nn.InstanceNorm1d(self.hidden_dim * 2, affine=True)

        # 改动3: 残差 shortcut 投影 — upstream 无此路径
        self.ts_shortcut = nn.Linear(self.input_seq_len, self.hidden_dim)

        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.hidden_dim)

        self.all_feat_dim = (self.hidden_dim + self.node_dim
                             + model_args['time_emb_dim'] * 2)

        # 改动1: 3-head 独立 Q/K 投影
        # upstream 只有 1 组 WQ, WK
        self.WQ_heads = nn.ModuleList([
            nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
            for _ in range(_NUM_HEADS)])
        self.WK_heads = nn.ModuleList([
            nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
            for _ in range(_NUM_HEADS)])

        # 改动4: 注意力 dropout
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X_in = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X_in.shape
        X_flat = X_in.view(B * N, S)

        # 时序嵌入 + InstanceNorm
        h1 = F.gelu(self.fc_ts_emb1(X_flat))
        # 改动2: InstanceNorm1d expects (N, C)
        h1 = self.norm(h1)
        h1 = self.dropout(h1)
        dy_feat = self.fc_ts_emb2(h1).view(B, N, -1)

        # 改动3: 残差 shortcut
        shortcut = self.ts_shortcut(X_flat).view(B, N, -1)
        dy_feat = dy_feat + shortcut

        _dbg(_TAG, "ts_feat", dy_feat=dy_feat, shortcut_norm=shortcut.norm())

        emb1 = E_d.unsqueeze(0).expand(B, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(B, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        feat_pair = [X1, X2]

        adjacent_list = []
        for feat in feat_pair:
            # 改动1: 3-head 注意力取平均
            head_weights = []
            for h in range(_NUM_HEADS):
                Q = self.WQ_heads[h](feat)
                K = self.WK_heads[h](feat)
                QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
                W = torch.softmax(QKT, dim=-1)
                # 改动4: attention dropout
                W = self.attn_drop(W)
                head_weights.append(W)

            # 多头平均
            W_avg = sum(head_weights) / _NUM_HEADS
            adjacent_list.append(W_avg)

            _dbg(_TAG, "multi_head_attn",
                 head_var=torch.stack(head_weights).var(dim=0).mean(),
                 W_avg=W_avg)

        return adjacent_list
