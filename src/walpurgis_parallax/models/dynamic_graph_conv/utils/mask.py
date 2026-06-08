"""
Mask — Parallax变体 (M054)
算法改动: REINFORCE重要性掩码 替代 确定性掩码
  原版: 直接用predefined adj做元素乘法
  Parallax: 用策略网络π(a|s)决定每条边是否保留
           s = (adj值, 节点度, 局部密度)
           a ∈ {keep, drop}
           reward = 下游loss的负值 (延迟到训练步结束)
           用REINFORCE梯度估计: ∇θ J = E[R * ∇θ log π(a|s)]
           训练时采样, 推理时用贪心策略(π > 0.5则保留)
           baseline用指数移动平均减方差

  策略梯度让掩码学会选择"真正重要的边", 而不是固定模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg, _reinforce_tracker


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 策略网络: 根据邻接值决定保留概率
        # 输入: 边权值 (scalar per edge)
        self.policy_fc1 = nn.Linear(1, 16)
        self.policy_fc2 = nn.Linear(16, 1)

        # REINFORCE baseline: EMA of rewards
        self.register_buffer(
            'reward_baseline', torch.tensor(0.0))
        self.baseline_momentum = model_args.get(
            'reinforce_baseline_momentum', 0.9)

        # 存储log_prob用于计算策略梯度
        self._log_probs = []
        self._entropy = 0.0

    def _policy_forward(self, edge_values):
        """策略网络: 边权值 → 保留概率"""
        # edge_values: [N, N] → [N, N, 1]
        x = edge_values.unsqueeze(-1)
        h = F.elu(self.policy_fc1(x))
        logits = self.policy_fc2(h).squeeze(-1)
        return logits

    def _reinforce_mask(self, index, adj):
        """REINFORCE策略梯度掩码"""
        base_mask = self.mask[index]
        # 策略网络产生保留概率
        policy_logits = self._policy_forward(base_mask)
        keep_prob = torch.sigmoid(policy_logits)
        keep_prob = torch.clamp(keep_prob, 1e-6, 1 - 1e-6)

        if self.training:
            # 采样: Bernoulli(keep_prob)
            action = torch.bernoulli(keep_prob)
            # 记录log π(a|s)用于后续REINFORCE更新
            log_p = (action * torch.log(keep_prob)
                     + (1 - action) * torch.log(1 - keep_prob))
            self._log_probs.append(log_p.sum())
            # 熵正则: 鼓励探索
            entropy = -(keep_prob * torch.log(keep_prob)
                        + (1 - keep_prob) * torch.log(1 - keep_prob))
            self._entropy += entropy.mean()
            mask_out = action
        else:
            # 推理: 贪心
            mask_out = (keep_prob > 0.5).float()

        result = mask_out.to(adj.device) * adj
        return result

    def get_reinforce_loss(self, reward):
        """在训练步结束后调用, 用reward计算策略梯度损失
        reward通常 = -task_loss (越低的loss给越高的reward)
        """
        if not self._log_probs:
            return torch.tensor(0.0)
        # 更新baseline
        baseline = self.reward_baseline.item()
        advantage = reward - baseline
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward)
        # REINFORCE损失: -advantage * Σ log π(a|s)
        total_log_prob = sum(self._log_probs)
        reinforce_loss = -advantage * total_log_prob
        # 加熵正则
        entropy_bonus = -0.01 * self._entropy

        _reinforce_tracker.record(
            reward, self._entropy.item()
            if isinstance(self._entropy, torch.Tensor) else 0.0)

        # 清空缓存
        self._log_probs = []
        self._entropy = 0.0

        return reinforce_loss + entropy_bonus

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._reinforce_mask(index, a))

        _dbg("mask.policy_active",
             f"training={self.training} "
             f"log_probs={len(self._log_probs)}", "graph")

        return result
