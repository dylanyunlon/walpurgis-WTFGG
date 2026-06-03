import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

_TAG = "mask"


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.base_mask = model_args['adjs']
        # 改动1: 可学习温度参数 τ, 控制 soft threshold 锐度
        # 初始 τ=1.0 (log_tau=0), 训练中自适应
        self.log_tau = nn.Parameter(torch.zeros(1))
        # 可学习阈值偏置
        self.threshold_bias = nn.Parameter(torch.tensor(-2.0))

    def _soft_mask(self, index, adj):
        base = self.base_mask[index].to(adj.device)

        # 改动1: softplus 阈值 — upstream 用硬乘法 mask * adj
        # 这里让 mask 值通过 softplus 生成 soft weight
        tau = torch.exp(self.log_tau).clamp(min=0.05, max=5.0)
        # soft threshold: sigmoid((base - bias) / tau)
        soft_w = torch.sigmoid((F.softplus(base) + self.threshold_bias) / tau)
        result = soft_w * adj

        # 改动2: 对角线清零 — upstream 未做此操作
        # 自环可能引入信息泄露, 显式去除
        n = result.shape[-1]
        if result.dim() == 2:
            result = result - torch.diag(torch.diag(result))
        elif result.dim() == 3:
            eye = torch.eye(n, device=result.device).unsqueeze(0)
            result = result * (1.0 - eye)

        _dbg(_TAG, f"soft_mask_{index}", tau=tau,
             soft_w_mean=soft_w.mean(), result_nnz=(result.abs() > 1e-6).sum())
        return result

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._soft_mask(index, a))
        return result
