"""
DistanceFunction — Nightfall变体
算法改写:
  1. attention距离: scaled-dot → 混合 cosine+dot 加权 (比例可学习)
  2. BatchNorm → LayerNorm (对小batch更稳定)
  3. 可学习attention temperature τ_attn
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_nightfall import _dbg


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        # 时序特征提取
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        # 时间槽embedding
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        # 距离分数: Q/K投影
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        # LayerNorm替代BatchNorm
        self.ln = nn.LayerNorm(self.hidden_dim * 2)
        # 可学习attention温度
        self.log_attn_tau = nn.Parameter(torch.zeros(1))
        # cosine-dot混合权重 (sigmoid输出)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        # LayerNorm替代BatchNorm
        h = F.relu(self.fc_ts_emb1(X))
        h = h.view(batch_size, num_nodes, -1)
        h = self.ln(h)
        h = h.view(batch_size * num_nodes, -1)
        dy_feat = self.fc_ts_emb2(self.dropout(h))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        _dbg("dist.dy_feat", dy_feat, "model")
        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        # attention温度
        tau = self.log_attn_tau.exp().clamp(min=0.1, max=10.0)
        # cosine-dot混合权重
        mix = torch.sigmoid(self.mix_logit)
        adjacent_list = []
        for feat in [X1, X2]:
            Q = self.WQ(feat)
            K = self.WK(feat)
            # 标准 scaled-dot attention
            dot_score = torch.bmm(Q, K.transpose(-1, -2)) / (math.sqrt(self.hidden_dim) * tau)
            # 余弦相似度 attention
            Q_norm = F.normalize(Q, dim=-1)
            K_norm = F.normalize(K, dim=-1)
            cos_score = torch.bmm(Q_norm, K_norm.transpose(-1, -2)) / tau
            # 混合
            combined = mix * cos_score + (1 - mix) * dot_score
            W = torch.softmax(combined, dim=-1)
            adjacent_list.append(W)
        _dbg("dist.attn_tau", tau, "model")
        _dbg("dist.mix_weight", mix, "model")
        return adjacent_list
