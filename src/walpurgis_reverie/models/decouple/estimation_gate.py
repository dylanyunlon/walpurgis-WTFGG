import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_reverie import _dbg

_TAG = "gate"


class EstimationGate(nn.Module):
    """upstream: FC→ReLU→FC→sigmoid gate on history_data
    改动: Bilinear fusion of node embeddings → Swish → FC → temperature-scaled sigmoid
    Bilinear捕获node_u/node_d之间的交互, Swish比ReLU平滑, temperature让gate更自适应
    """

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        # 改动1: 用bilinear而非concat+linear来融合两个node embedding
        self.bilinear = nn.Bilinear(node_emb_dim, node_emb_dim, hidden_dim)
        # 改动2: 时间特征单独投影
        self.time_proj = nn.Linear(time_emb_dim * 2, hidden_dim)
        # 改动3: 融合后用SiLU (Swish)代替ReLU
        self.activation = nn.SiLU()
        self.fc_out = nn.Linear(hidden_dim, 1)
        # 改动4: 可学习温度参数控制sigmoid陡度
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, num_nodes, _ = time_in_day_feat.shape

        # expand node embeddings to (B, L, N, d)
        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_length, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_length, -1, -1)

        # bilinear融合: 捕获u/d之间的二阶交互
        bilinear_feat = self.bilinear(
            eu.reshape(-1, eu.size(-1)),
            ed.reshape(-1, ed.size(-1))
        ).view(batch_size, seq_length, num_nodes, -1)

        # 时间特征投影
        time_cat = torch.cat([time_in_day_feat, day_in_week_feat], dim=-1)
        time_feat = self.time_proj(time_cat)

        # 融合 + Swish
        hidden = self.activation(bilinear_feat + time_feat)

        # temperature-scaled sigmoid
        gate = torch.sigmoid(
            self.fc_out(hidden) / (self.temperature.abs() + 0.1)
        )
        gate = gate[:, -history_data.shape[1]:, :, :]

        _dbg(f"{_TAG}/gate_val", gate, _TAG)
        _dbg(f"{_TAG}/temperature", self.temperature, _TAG)

        return history_data * gate
