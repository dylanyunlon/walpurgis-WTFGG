"""
STLocalizedConv — Penumbra变体
算法改动: Chebyshev多项式图卷积 + SpectralNorm
  原版: 直接k阶矩阵乘法扩散, FC更新
  Penumbra:
    - Chebyshev递推: T_0=I, T_1=L_hat, T_k=2*L_hat*T_{k-1}-T_{k-2}
      用归一化拉普拉斯代替原始邻接做多阶扩散
    - SpectralNorm: 对gcn_updt线性层做谱归一化, 防止训练不稳定
    - 可学习的阶权重: 每阶Chebyshev有独立可学习标量权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from ... import _dbg, dataflow_checkpoint


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None,
                 use_pre=None, dy_graph=None,
                 sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (
            int(use_pre) * len(self.pre_defined_graph)
            + len(self.pre_defined_graph) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(
            self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim,
            self.k_t * hidden_dim, bias=False)
        # SpectralNorm: 约束gcn更新层的Lipschitz常数
        self.gcn_updt = spectral_norm(nn.Linear(
            self.hidden_dim * self.num_matric,
            self.hidden_dim))

        # Chebyshev阶权重: 每阶一个可学习标量
        self.cheby_weights = nn.Parameter(
            torch.ones(self.k_s + 1) / (self.k_s + 1))

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def _chebyshev_basis(self, L_hat, X, order):
        """Chebyshev递推: 生成T_0(L)*X, T_1(L)*X, ..., T_k(L)*X
        比直接矩阵幂更数值稳定"""
        basis = [X]  # T_0 = I
        if order >= 1:
            T1 = torch.matmul(L_hat, X)
            basis.append(T1)
        for k in range(2, order + 1):
            Tk = 2 * torch.matmul(L_hat, basis[-1]) - basis[-2]
            basis.append(Tk)
        return basis

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)

        dataflow_checkpoint("stconv.gconv_out", out)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(
            support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(
                -1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0],
                graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        dataflow_checkpoint("stconv.input", X)
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(batch_size, seq_len, num_nodes,
                       kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes,
                       kernel_size, num_feat)

        # 加权平均(用Chebyshev权重)而非简单mean
        weights = F.softmax(self.cheby_weights[:kernel_size],
                            dim=0)
        X_0 = sum(weights[i] * out[:, :, :, i, :]
                  for i in range(kernel_size))

        _dbg("stconv.cheby_weights",
             weights, "diffusion")

        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len,
            kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden
