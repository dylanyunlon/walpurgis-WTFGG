"""
dif_model.py — CardGame STLocalizedConv
算法改写 (vs upstream):
  - BatchNorm + ReLU → WeightNorm + Mish激活
  - gconv中新增残差skip connection: out = gconv(x) + x_residual
  - fc_list_updt使用WeightNorm而非普通Linear
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="DifModel"):
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


def mish(x):
    """Mish激活: x * tanh(softplus(x))"""
    return x * torch.tanh(F.softplus(x))


class STLocalizedConv(nn.Module):
    """CardGame ST Localized Conv with WeightNorm + Mish + gconv residual skip"""

    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None,
                 dy_graph=None, sta_graph=None, **model_args):
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
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        # WeightNorm替代普通Linear
        self.fc_list_updt = nn.utils.parametrizations.weight_norm(
            nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False))
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)

        # 残差skip projection (如果维度不匹配)
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim) if True else None

        # Mish替代BN+ReLU (不需要BN了)

    def gconv(self, support, X_k, X_0):
        """图卷积 + 残差skip"""
        residual = X_0  # 保存残差
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
        # 残差skip connection
        out = out + self.skip_proj(residual)
        _dbg("gconv.out+skip", out)
        return out

    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        _dbg("input.X", X)
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        # Mish激活替代ReLU
        out = mish(out)
        _dbg("after_mish", out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len, kernel_size * num_nodes, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        _dbg("output.hidden", hidden)
        return hidden
