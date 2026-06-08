"""
DynamicGraphConstructor — Aurora变体
算法改写: 自适应图正则化 (Spectral Graph Regularizer)
  - 在forward中计算学到的图的拉普拉斯矩阵
  - 约束其特征值分布, 使图结构更稳定
  - 正则化损失通过graph_reg_loss属性暴露给trainer
"""
import torch
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder


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

        # Aurora: spectral regularizer — 目标特征值分布参数
        # 鼓励图的拉普拉斯特征值集中在[0, 2]范围内, 避免过度连接或孤立
        self.target_spectral_radius = nn.Parameter(
            torch.tensor(1.5), requires_grad=False)
        # 存储最近一次forward的正则化损失
        self.graph_reg_loss = torch.tensor(0.0)

    def _compute_spectral_reg(self, adj_list):
        """计算图拉普拉斯的spectral正则化损失

        对每个邻接矩阵:
        1. 计算度矩阵D和拉普拉斯L = D - A
        2. 用Frobenius范数近似约束特征值分布
        3. 惩罚过大的spectral radius (最大特征值)
        """
        reg = torch.tensor(0.0, device=adj_list[0].device)
        count = 0
        for adj in adj_list:
            if adj.dim() == 3:
                # batched adj: [B, N, N]
                # 计算对称化的拉普拉斯
                adj_sym = (adj + adj.transpose(-1, -2)) / 2.0
                degree = adj_sym.sum(dim=-1)  # [B, N]
                # L = D - A (unnormalized Laplacian)
                D = torch.diag_embed(degree)  # [B, N, N]
                L = D - adj_sym

                # Frobenius范数作为特征值能量的代理
                # ||L||_F^2 = sum(eigenvalues^2)
                frob_sq = (L ** 2).sum(dim=(-1, -2)).mean()

                # 目标: frob_sq应该与N * target^2成比例
                N = adj.shape[-1]
                target_energy = N * (self.target_spectral_radius ** 2)
                reg = reg + (frob_sq - target_energy) ** 2 / (target_energy ** 2 + 1e-6)
                count += 1
        if count > 0:
            reg = reg / count
        return reg

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0],
                    k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)

        # Aurora: 计算spectral正则化损失
        self.graph_reg_loss = self._compute_spectral_reg(dist_mx)

        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)

        return dynamic_graphs
