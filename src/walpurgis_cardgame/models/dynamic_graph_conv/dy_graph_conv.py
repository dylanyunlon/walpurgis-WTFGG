"""
dy_graph_conv.py — CardGame DynamicGraphConstructor
算法改写 (vs upstream):
  - 新增DropNode正则化: 训练时随机丢弃节点
  - 新增可学习接收增益: 每个节点一个可学习缩放因子
"""
import os
import sys
import torch
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="DyGraphConv"):
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


class DropNode(nn.Module):
    """DropNode正则化: 训练时随机mask掉一定比例的节点"""

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, graph):
        """graph: [B, N, ...]"""
        if not self.training or self.p == 0:
            return graph
        num_nodes = graph.shape[-1] if graph.dim() == 3 else graph.shape[0]
        mask = torch.bernoulli(
            torch.ones(num_nodes, device=graph.device) * (1 - self.p))
        # 对行和列同时mask
        if graph.dim() == 3:
            # [B, N, N]
            row_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            col_mask = mask.unsqueeze(0).unsqueeze(1)   # [1, 1, N]
            graph = graph * row_mask * col_mask / ((1 - self.p) ** 2)
        else:
            row_mask = mask.unsqueeze(-1)
            col_mask = mask.unsqueeze(0)
            graph = graph * row_mask * col_mask / ((1 - self.p) ** 2)
        return graph


class DynamicGraphConstructor(nn.Module):
    """CardGame Dynamic Graph with DropNode + learnable receive gain"""

    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.num_nodes = model_args['num_nodes']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        # CardGame新增: DropNode + learnable receive gain
        self.drop_node = DropNode(p=model_args.get('dropout', 0.1) * 0.5)
        self.receive_gain = nn.Parameter(
            torch.ones(self.num_nodes))

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        _dbg("input.X", X)

        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        _dbg("dist_mx[0]", dist_mx[0])

        # mask
        dist_mx = self.mask(dist_mx)

        # CardGame: DropNode正则化
        dist_mx = [self.drop_node(d) for d in dist_mx]

        # normalization
        dist_mx = self.normalizer(dist_mx)

        # CardGame: 可学习接收增益 (对每个节点的接收边加权)
        gain = torch.sigmoid(self.receive_gain)
        dist_mx = [d * gain.unsqueeze(0).unsqueeze(-1) for d in dist_mx]
        _dbg("receive_gain", gain, module="DyGraphConv")

        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)

        _dbg("dynamic_graphs.len", len(dynamic_graphs))
        return dynamic_graphs
