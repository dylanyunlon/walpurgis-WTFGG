import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

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
        # 这里让每个时间步有独立权重, softmax 归一化
        self.temporal_logits = nn.Parameter(torch.zeros(self.k_t))

        # 改动2: cosine-similarity 辅助投影
        self.cos_proj = nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=False)

    def st_localization(self, graph_ordered):
        # 改动1: 取 softmax 时间权重
        t_weights = F.softmax(self.temporal_logits, dim=0)

        _dbg(_TAG, "temporal_weights",
             weights=t_weights, logits=self.temporal_logits)

        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                # expand 到 k_t 维
                g_exp = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1)
                # 改动1: 用可学习权重加权每个时间步
                # upstream 直接 expand 不加权, 所有时间步等权
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

        # 改动2: cosine-similarity 辅助 —
        # 在 dist_mx 上加一个 cosine similarity 项作为修正
        B = X.shape[0]
        ed_proj = self.cos_proj(E_d.unsqueeze(0).expand(B, -1, -1))
        eu_proj = self.cos_proj(E_u.unsqueeze(0).expand(B, -1, -1))
        cos_sim = F.cosine_similarity(
            ed_proj.unsqueeze(2), eu_proj.unsqueeze(1), dim=-1)
        # 归一化到 [0, 1] 范围
        cos_sim = (cos_sim + 1.0) * 0.5

        for i in range(len(dist_mx)):
            dist_mx[i] = dist_mx[i] + 0.1 * cos_sim

        _dbg(_TAG, "cos_augment", cos_sim_mean=cos_sim.mean())

        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)

        _dbg(_TAG, "output", n_graphs=len(dynamic_graphs))

        # 改动3: 梯度范数监控 (只在训练时有意义)
        if self.training and self.temporal_logits.grad is not None:
            _dbg(_TAG, "grad_check",
                 temporal_grad_norm=self.temporal_logits.grad.norm())

        return dynamic_graphs
