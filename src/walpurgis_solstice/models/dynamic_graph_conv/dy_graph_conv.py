import torch
import torch.nn as nn
import sys, os

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:dygraph:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} nnz_frac={((val.abs()>1e-6).float().mean().item()):.2%}", file=sys.stderr)

class DynamicGraphConstructor(nn.Module):
    """solstice: 增加可学习时间权重对k_t步加权"""
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
        # solstice: 可学习时间权重 (softmax归一化)
        self.temporal_logits = nn.Parameter(torch.zeros(self.k_t))

    def st_localization(self, graph_ordered):
        # solstice: softmax时间权重
        t_weights = torch.softmax(self.temporal_logits, dim=0)
        _adbg("temporal_weights", t_weights)
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                # solstice: 用可学习权重加权每个时间步
                w = t_weights.view(1, 1, self.k_t, 1).to(k_order_graph.device)
                k_order_graph = k_order_graph * w
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
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        _adbg("dynamic_graph_count", len(dynamic_graphs))
        return dynamic_graphs
