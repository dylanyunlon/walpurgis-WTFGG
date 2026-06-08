"""
Normalizer — Parallax变体 (M054)
算法改动: Spectral Clustering归一化 替代 单行归一化
  原版: D^{-1} * A (行归一化)
  Parallax: 先对邻接矩阵做轻量谱聚类(特征分解找簇)
           然后在每个簇内做独立归一化
           不同簇之间的边权用簇间系数缩放
           这让同一社区内的信息传播更充分,
           社区间的传播有衰减, 更符合图的社区结构

  实现: 用邻接矩阵的Laplacian的前k个特征向量做谱嵌入
       → k-means聚类 → 簇内行归一化 + 簇间缩放
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    def __init__(self, n_clusters=3, eps=1e-6):
        super().__init__()
        self.n_clusters = n_clusters
        self.eps = eps
        # 簇间缩放系数: 学习不同簇之间的交互强度
        self.inter_cluster_scale = nn.Parameter(
            torch.tensor(0.3))
        # 簇内增强系数
        self.intra_cluster_boost = nn.Parameter(
            torch.tensor(1.2))

    def _spectral_partition(self, adj):
        """轻量谱分区: 用邻接矩阵的特征向量做软聚类
        返回每个节点的簇membership (软分配)
        adj: [N, N] (单个图)
        """
        N = adj.shape[-1]
        n_clusters = min(self.n_clusters, N)

        # 度矩阵
        deg = adj.sum(dim=-1) + self.eps
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        # 归一化拉普拉斯: I - D^{-1/2} A D^{-1/2}
        L_norm = torch.eye(N, device=adj.device) - \
            D_inv_sqrt @ adj @ D_inv_sqrt
        L_norm = _remove_nan_inf(L_norm)

        # 取前k个最小特征值对应的特征向量
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
            # 前k个特征向量(最小特征值)
            spectral_emb = eigenvectors[:, :n_clusters]
        except Exception:
            # fallback: 均匀分配
            spectral_emb = torch.randn(
                N, n_clusters, device=adj.device) * 0.1

        # 软聚类: softmax over spectral embeddings
        # 每个节点属于每个簇的概率
        cluster_logits = spectral_emb / (
            spectral_emb.norm(dim=-1, keepdim=True) + self.eps)
        membership = F.softmax(
            cluster_logits * 5.0, dim=-1)  # 温度=0.2
        return membership

    def _cluster_aware_norm(self, graph):
        """谱聚类感知的归一化"""
        if len(graph.shape) == 2:
            # [N, N]
            membership = self._spectral_partition(graph)
            return self._apply_cluster_norm(
                graph, membership)
        else:
            # [B, N, N]
            results = []
            for b in range(graph.shape[0]):
                membership = self._spectral_partition(
                    graph[b])
                normed = self._apply_cluster_norm(
                    graph[b], membership)
                results.append(normed)
            return torch.stack(results, dim=0)

    def _apply_cluster_norm(self, graph, membership):
        """对单个图应用簇感知归一化
        graph: [N, N]
        membership: [N, K] 软聚类分配
        """
        N = graph.shape[0]
        K = membership.shape[1]
        intra_scale = torch.sigmoid(self.intra_cluster_boost)
        inter_scale = torch.sigmoid(self.inter_cluster_scale)

        # 计算节点对之间的"同簇概率"
        # P(i,j同簇) = Σ_k membership[i,k] * membership[j,k]
        same_cluster_prob = torch.mm(
            membership, membership.T)  # [N, N]
        same_cluster_prob = torch.clamp(
            same_cluster_prob, 0, 1)

        # 簇内边增强, 簇间边缩放
        scale_matrix = (
            same_cluster_prob * intra_scale
            + (1 - same_cluster_prob) * inter_scale
        )
        scaled_graph = graph * scale_matrix

        # 行归一化
        row_sum = scaled_graph.sum(dim=-1, keepdim=True) + self.eps
        normed = scaled_graph / row_sum

        normed = _remove_nan_inf(normed)
        return normed

    def forward(self, adj):
        normed = [self._cluster_aware_norm(a) for a in adj]
        _dbg("normalizer.inter_scale",
             torch.sigmoid(self.inter_cluster_scale), "graph")
        _dbg("normalizer.intra_boost",
             torch.sigmoid(self.intra_cluster_boost), "graph")
        return normed


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
        return [self._multi_order(a) for a in adj]
