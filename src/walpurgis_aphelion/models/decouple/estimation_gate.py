"""
Aphelion EstimationGate — 算法改写 #1:
  upstream: FC1 → ReLU → FC2 → sigmoid → element-wise multiply
  corona:  Attention Pooling (Q/K from node+time embed) → SiLU → learned bias → sigmoid
  aphelion: Hypernetwork gate — 一个小型条件网络根据时空嵌入动态生成gate权重,
            而非使用固定的FC层。hypernetwork的输出是主gate网络的参数,
            实现输入自适应的gate机制
  改动幅度: ~25% (hypernetwork替代固定FC权重)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        feat_dim = 2 * node_emb_dim + time_emb_dim * 2
        # Aphelion改写: hypernetwork — 用小网络动态生成gate的权重
        # 而非upstream/corona的固定FC层
        # hypernet接收聚合的上下文特征, 输出主gate网络的参数
        self.context_dim = hidden_dim  # hypernetwork的瓶颈维度
        # 上下文压缩: feat_dim → context_dim
        self.context_compressor = nn.Sequential(
            nn.Linear(feat_dim, self.context_dim),
            nn.GELU(),  # Aphelion: GELU替代ReLU/SiLU
        )
        # Hypernetwork: 生成主gate网络的权重 (context_dim → feat_dim)
        # 主gate: feat → hidden → 1, 所以需要生成两组权重
        self.hyper_w1 = nn.Linear(self.context_dim, feat_dim * hidden_dim)
        self.hyper_b1 = nn.Linear(self.context_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(self.context_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(self.context_dim, 1)
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        num_nodes = node_embedding_u.shape[0]
        estimation_gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)

        dataflow_checkpoint("aphelion_gate.feat", estimation_gate_feat)

        # Aphelion: hypernetwork — 先压缩上下文, 再动态生成gate权重
        # 取时间维均值作为全局上下文 [B, N, feat_dim] → [B, N, context_dim]
        ctx_input = estimation_gate_feat.mean(dim=1)  # [B, N, feat_dim]
        ctx = self.context_compressor(ctx_input)  # [B, N, context_dim]

        # 动态生成主gate网络的权重
        w1 = self.hyper_w1(ctx).view(batch_size, num_nodes, self.feat_dim, self.hidden_dim)
        b1 = self.hyper_b1(ctx)  # [B, N, hidden_dim]
        w2 = self.hyper_w2(ctx)  # [B, N, hidden_dim]
        b2 = self.hyper_b2(ctx)  # [B, N, 1]

        # 用动态权重执行主gate前向: feat → hidden → gate
        # estimation_gate_feat: [B, S, N, feat_dim]
        x = estimation_gate_feat  # [B, S, N, feat_dim]
        # 对每个(b,n)用对应的动态权重做线性变换
        # x: [B, S, N, feat_dim], w1: [B, N, feat_dim, hidden_dim]
        h = torch.einsum('bsnf,bnfh->bsnh', x, w1) + b1.unsqueeze(1)
        h = F.gelu(h)  # Aphelion: GELU激活
        # w2: [B, N, hidden_dim] → 相当于每个位置做点积
        gate_logit = (h * w2.unsqueeze(1)).sum(dim=-1, keepdim=True) + b2.unsqueeze(1)
        estimation_gate = torch.sigmoid(gate_logit)[:, -history_data.shape[1]:, :, :]

        dataflow_checkpoint("aphelion_gate.output", estimation_gate)
        history_data = history_data * estimation_gate
        return history_data
