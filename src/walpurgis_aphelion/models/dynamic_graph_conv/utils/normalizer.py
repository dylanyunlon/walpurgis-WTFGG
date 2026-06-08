"""
Aphelion Normalizer — 算法改写 #6:
  upstream: D^{-1}A (行归一化)
  corona: 交替归一化 D_row^{-1/2} A D_col^{-1/2}
  aphelion: Wavelet-based normalizer — 使用简化的图小波变换进行归一化。
            先将图矩阵用切比雪夫多项式近似的图小波基展开,
            在小波域做归一化(各频段独立归一化), 再逆变换回来。
            比单纯的度归一化保留更多频率信息。
  改动幅度: ~35% (小波域归一化替代度矩阵归一化)
"""
import torch
import torch.nn as nn


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Aphelion: 小波归一化的可学习缩放因子 (每个频段一个)
        self.num_wavelet_scales = 3  # 使用3个尺度的小波
        self.scale_weights = nn.Parameter(torch.ones(self.num_wavelet_scales))

    def _chebyshev_basis(self, graph, order):
        """计算图矩阵的切比雪夫多项式基 T_0, T_1, ..., T_{order-1}"""
        # T_0 = I, T_1 = L_scaled, T_k = 2*L*T_{k-1} - T_{k-2}
        N = graph.shape[-1]
        # 归一化拉普拉斯: L = I - D^{-1/2} A D^{-1/2}, 缩放到[-1,1]
        row_sum = torch.sum(graph, dim=-1)
        d_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(row_sum + 1e-8))
        if graph.dim() == 3:
            D = torch.diag_embed(d_inv_sqrt)
            L = torch.eye(N, device=graph.device).unsqueeze(0) - torch.bmm(torch.bmm(D, graph), D)
        else:
            D = torch.diag(d_inv_sqrt)
            L = torch.eye(N, device=graph.device) - D @ graph @ D
        # 缩放到[-1, 1]: L_scaled = 2*L/lambda_max - I ≈ 2*L - I (lambda_max≈2)
        L_scaled = 2.0 * L - torch.eye(N, device=graph.device).unsqueeze(0) if graph.dim() == 3 \
            else 2.0 * L - torch.eye(N, device=graph.device)

        bases = []
        if graph.dim() == 3:
            T0 = torch.eye(N, device=graph.device).unsqueeze(0).expand_as(graph)
        else:
            T0 = torch.eye(N, device=graph.device)
        bases.append(T0)
        if order > 1:
            T1 = L_scaled
            bases.append(T1)
        for k in range(2, order):
            Tk = 2.0 * torch.matmul(L_scaled, bases[-1]) - bases[-2]
            bases.append(Tk)
        return bases

    def _norm(self, graph):
        # Aphelion: wavelet-based归一化
        # 1. 计算切比雪夫基 (近似图小波)
        bases = self._chebyshev_basis(graph, self.num_wavelet_scales)

        # 2. 在小波域做归一化: 对每个尺度的分量独立归一化
        weights = torch.softmax(self.scale_weights, dim=0)
        normed = torch.zeros_like(graph)
        for k, basis in enumerate(bases):
            # 小波系数: graph在第k个基上的投影
            coeff = torch.matmul(basis, graph)
            # 归一化系数
            coeff_norm = remove_nan_inf(coeff / (coeff.abs().sum(dim=-1, keepdim=True) + 1e-8))
            # 加权重建
            normed = normed + weights[k] * torch.matmul(basis.transpose(-1, -2), coeff_norm)

        # 确保非负
        normed = torch.relu(normed)
        # 最终行归一化确保随机矩阵性质
        row_sum = normed.sum(dim=-1, keepdim=True) + 1e-8
        normed = normed / row_sum
        return remove_nan_inf(normed)

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
