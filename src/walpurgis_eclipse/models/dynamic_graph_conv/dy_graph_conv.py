"""Eclipse dynamic graph constructor."""
import torch.nn as nn, sys, os
from .utils import DistanceFunction, Mask, Normalizer, MultiOrder
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']; self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']; self.node_dim = model_args['node_hidden']
        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args); self.normalizer = Normalizer(); self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local = []
        for modality_i in graph_ordered:
            for kg in modality_i:
                kg = kg.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                st_local.append(kg.reshape(kg.shape[0], kg.shape[1], kg.shape[2] * kg.shape[3]))
        return st_local

    def forward(self, **inputs):
        dist_mx = self.distance_function(inputs['history_data'], inputs['node_embedding_d'], inputs['node_embedding_u'], inputs['time_in_day_feat'], inputs['day_in_week_feat'])
        dist_mx = self.mask(dist_mx); dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dg = self.st_localization(mul_mx)
        if _ECL_DBG: print(f"[ECL:dygraph] n_graphs={len(dg)} sparsity={(dg[0]==0).float().mean().item():.2%}", file=sys.stderr)
        return dg
