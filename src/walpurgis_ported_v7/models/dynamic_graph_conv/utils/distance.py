import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

_DBG_DIST = ("--dbg-dist" in sys.argv)


class DistanceFunction(nn.Module):
    """算法改动:
    1. 在 ts embedding 阶段加 residual connection (fc2 输出 + 原始均值池化)
    2. QKT 计算时加 temperature parameter (可学习), 替代固定的 sqrt(d) 缩放
    """
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

        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = (self.ts_feat_dim + self.node_dim
                            + model_args['time_emb_dim'] * 2)
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # 算法改动: 可学习的 temperature 参数
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(self.hidden_dim)))

        # 算法改动: residual projection for ts embedding
        self.ts_residual_proj = nn.Linear(self.input_seq_len, self.hidden_dim)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X_raw = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X_raw.shape
        X_flat = X_raw.view(batch_size * num_nodes, seq_len)

        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.relu(self.fc_ts_emb1(X_flat)))))

        # 算法改动: residual — 原始序列的线性投影 + MLP 输出
        residual = self.ts_residual_proj(X_flat)
        dy_feat = dy_feat + residual * 0.3

        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        if _DBG_DIST:
            with torch.no_grad():
                print(f"[DBG-DIST] dy_feat shape={list(dy_feat.shape)}  "
                      f"norm={dy_feat.norm().item():.4f}  "
                      f"residual_ratio={residual.norm().item() / (dy_feat.norm().item() + 1e-8):.4f}")

        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X_list = [X1, X2]

        adjacent_list = []
        for feat in X_list:
            Q = self.WQ(feat)
            K = self.WK(feat)
            # 算法改动: 用可学习 temperature 代替 固定 sqrt(d)
            temp = torch.clamp(self.temperature, min=1.0)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / temp
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)

            if _DBG_DIST:
                with torch.no_grad():
                    print(f"[DBG-DIST] temp={temp.item():.3f}  "
                          f"adj_sparsity={(W < 0.01).float().mean().item():.3f}  "
                          f"adj_max={W.max().item():.5f}")

        return adjacent_list
