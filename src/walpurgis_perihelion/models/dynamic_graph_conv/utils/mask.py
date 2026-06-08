"""
Mask — Perihelion变体
算法改动: Gumbel-Softmax离散化掩码(温度退火)
  原版: 直接用predefined adj做元素乘法
  Perihelion: 将掩码视为离散选择问题(保留/丢弃边)
             用Gumbel-Softmax做可微松弛, 温度退火控制硬度
             τ_init=2.0 → τ_min=0.1 (线性退火)
             训练时soft, 推理时hard argmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg, _gumbel_tracker


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Gumbel-Softmax温度: 初始值较高(soft), 逐步退火(hard)
        self.tau_init = 2.0
        self.tau_min = 0.1
        self.register_buffer('current_tau',
                             torch.tensor(self.tau_init))
        self.register_buffer('anneal_step',
                             torch.tensor(0, dtype=torch.long))
        self.total_anneal_steps = 5000

    def anneal_temperature(self):
        """线性温度退火: 每步调用一次"""
        progress = min(
            self.anneal_step.item() / max(self.total_anneal_steps, 1),
            1.0)
        new_tau = self.tau_init - (self.tau_init - self.tau_min) * progress
        self.current_tau.fill_(max(new_tau, self.tau_min))
        self.anneal_step += 1

    def _gumbel_softmax_mask(self, index, adj):
        """Gumbel-Softmax离散化掩码"""
        base_mask = self.mask[index]
        # 转换为logits: log(p / (1-p))
        prob = torch.clamp(base_mask, 1e-6, 1.0 - 1e-6)
        logit_keep = torch.log(prob)
        logit_drop = torch.log(1.0 - prob)
        # 组合成2-类logits: [keep, drop]
        logits = torch.stack([logit_keep, logit_drop], dim=-1)

        if self.training:
            self.anneal_temperature()
            tau = self.current_tau.item()
            # Gumbel-Softmax采样
            gumbel_out = F.gumbel_softmax(
                logits, tau=tau, hard=False, dim=-1)
            # 取keep概率
            soft_mask = gumbel_out[..., 0]

            # 计算掩码熵用于诊断
            with torch.no_grad():
                entropy = -(soft_mask * torch.log(soft_mask + 1e-8)
                            + (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8))
                _gumbel_tracker.record(tau, entropy.mean().item())
        else:
            # 推理: hard argmax
            soft_mask = (prob > 0.5).float()

        return soft_mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._gumbel_softmax_mask(index, a))

        _dbg("mask.gumbel_tau",
             self.current_tau, "graph")

        return result
