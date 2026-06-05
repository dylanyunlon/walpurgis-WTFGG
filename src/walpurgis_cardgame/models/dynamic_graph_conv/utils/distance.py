"""
distance.py — CardGame DistanceFunction
算法改写 (vs upstream):
  - scaled-dot attention → Mahalanobis距离 + 可学习协方差
  - 引入可学习对角协方差矩阵Σ, 计算 d(q,k) = (q-k)^T Σ^{-1} (q-k)
  - 使用对角Σ的log参数化保证正定性
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="Distance"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class DistanceFunction(nn.Module):
    """CardGame Distance: Mahalanobis距离 + 可学习对角协方差"""

    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']

        # Time Series Feature Extraction
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim

        # Time Slot Embedding
        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        # Feature projection
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # CardGame: 可学习Mahalanobis对角协方差 (log参数化保证正定)
        self.log_diag_cov = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, X, E_d, E_u, T_D, D_W):
        _dbg("input.X", X)
        # last pooling
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        # feature concatenation
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X_list = [X1, X2]

        # CardGame: Mahalanobis距离 + 可学习协方差
        inv_cov_diag = torch.exp(-self.log_diag_cov)  # Σ^{-1}对角元素
        _dbg("inv_cov_diag", inv_cov_diag)

        adjacent_list = []
        for feat in X_list:
            Q = self.WQ(feat)  # [B, N, D]
            K = self.WK(feat)  # [B, N, D]

            # Mahalanobis: d(i,j) = (Qi - Kj)^T diag(σ^{-2}) (Qi - Kj)
            # 展开: Q^T Σ^{-1} Q + K^T Σ^{-1} K - 2 Q^T Σ^{-1} K
            Q_scaled = Q * inv_cov_diag.unsqueeze(0).unsqueeze(0)
            QQ = (Q_scaled * Q).sum(-1, keepdim=True)   # [B, N, 1]
            KK = (K * inv_cov_diag.unsqueeze(0).unsqueeze(0) * K).sum(-1, keepdim=True)  # [B, N, 1]
            QK = torch.bmm(Q_scaled, K.transpose(-1, -2))  # [B, N, N]

            dist = QQ + KK.transpose(-1, -2) - 2 * QK  # [B, N, N]
            # 负距离做softmax得到相似度
            W = torch.softmax(-dist / math.sqrt(self.hidden_dim), dim=-1)
            _dbg("mahalanobis_adj", W)
            adjacent_list.append(W)

        return adjacent_list
