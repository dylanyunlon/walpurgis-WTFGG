import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis import _dbg

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_TAG = "dygraph"


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        # 改动1: 可学习时间权重 — upstream 对 k_t 步均匀 expand
        self.temporal_logits = nn.Parameter(torch.zeros(self.k_t))

        # 改动2: cosine-similarity 辅助投影
        self.cos_proj = nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=False)

        # 改动3: 可学习 cosine 混合系数 — upstream 无此路径
        # sigmoid(raw_alpha) 控制 cos_sim 对图的修正强度, init ≈ 0.1
        self._raw_cos_alpha = nn.Parameter(torch.tensor(-2.2))

        # DropEdge 正则 — 训练时随机置零一部分边
        # 余弦退火策略: 前期高 drop 帮助正则化, 后期低 drop 精细收敛
        self._edge_drop_base = model_args.get('dropout', 0.1) * 0.5
        self._edge_drop_min = self._edge_drop_base * 0.1  # 最低衰减到 base 的 10%
        self._drop_anneal_steps = 5000  # 经过多少步完成一个余弦周期
        self._current_step = 0

        # 改动5: 边重要性缩放 — 对每条边乘以 softplus(learnable_bias)
        n_nodes = model_args['num_nodes']
        self.edge_scale = nn.Parameter(torch.zeros(n_nodes))

    def _drop_edges(self, adj):
        """训练时随机丢弃 adj 中的边, drop rate 按余弦退火."""
        if not self.training or self._edge_drop_base <= 0:
            return adj
        # 余弦退火: rate 从 base 衰减到 min
        import math
        progress = min(self._current_step / max(self._drop_anneal_steps, 1), 1.0)
        rate = self._edge_drop_min + 0.5 * (self._edge_drop_base - self._edge_drop_min) * (
            1.0 + math.cos(math.pi * progress))
        self._current_step += 1
        mask = torch.bernoulli(torch.full_like(adj, 1.0 - rate))
        return adj * mask / max(1.0 - rate, 1e-6)

    def st_localization(self, graph_ordered):
        t_weights = F.softmax(self.temporal_logits, dim=0)

        _dbg(_TAG, "temporal_weights",
             weights=t_weights, logits=self.temporal_logits)

        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                g_exp = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1)
                # 改动1: 可学习时间权重加权
                w = t_weights.view(1, 1, self.k_t, 1)
                g_exp = g_exp * w
                g_exp = g_exp.reshape(
                    g_exp.shape[0], g_exp.shape[1],
                    g_exp.shape[2] * g_exp.shape[3])
                st_local_graph.append(g_exp)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        _dbg(_TAG, "input", X=X, E_d=E_d, E_u=E_u)

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)

        # 改动2+3: cosine-similarity 辅助, 可学习混合系数
        B = X.shape[0]
        ed_proj = self.cos_proj(E_d.unsqueeze(0).expand(B, -1, -1))
        eu_proj = self.cos_proj(E_u.unsqueeze(0).expand(B, -1, -1))
        cos_sim = F.cosine_similarity(
            ed_proj.unsqueeze(2), eu_proj.unsqueeze(1), dim=-1)
        cos_sim = (cos_sim + 1.0) * 0.5

        cos_alpha = torch.sigmoid(self._raw_cos_alpha)
        for i in range(len(dist_mx)):
            dist_mx[i] = dist_mx[i] + cos_alpha * cos_sim

        _dbg(_TAG, "cos_augment",
             cos_alpha=cos_alpha, cos_sim_mean=cos_sim.mean())

        # 改动4: DropEdge
        dist_mx = [self._drop_edges(a) for a in dist_mx]

        # 改动5: 边重要性缩放 — 每个目标节点有独立的接收增益
        # scale_j 控制节点 j 接收多少邻居信息
        scale = F.softplus(self.edge_scale)     # 保证 > 0
        for i in range(len(dist_mx)):
            # dist_mx[i]: (B, N, N), scale: (N,) → 广播到列维
            dist_mx[i] = dist_mx[i] * scale.unsqueeze(0).unsqueeze(0)

        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)

        _dbg(_TAG, "output", n_graphs=len(dynamic_graphs),
             edge_scale_mean=scale.mean(),
             edge_scale_std=scale.std())

        return dynamic_graphs
