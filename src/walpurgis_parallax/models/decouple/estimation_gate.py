"""
EstimationGate — Parallax变体 (M054)
算法改动: Bayesian MC-Dropout Gate
  原版: 2层FC → sigmoid 生成逐时间步门控
  Parallax: 多次前向传播(MC dropout采样), 计算预测均值和方差
           不确定性高的时间步门控值更保守(接近0.5)
           不确定性低的时间步允许更极端的门控
           推理时做T次采样取均值, 训练时单次采样+正则

  核心思想: 门控决策本身有不确定性, 用贝叶斯方法量化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint, _mc_dropout_tracker


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim,
                 mc_samples=5, dropout_rate=0.2):
        super().__init__()
        input_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate

        # 主通道: 2层FC (与dropout配合做MC采样)
        self.fc_gate_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_gate_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_gate_out = nn.Linear(hidden_dim // 2, 1)

        # MC-Dropout层: 训练和推理时都开启
        self.mc_drop_1 = nn.Dropout(p=dropout_rate)
        self.mc_drop_2 = nn.Dropout(p=dropout_rate)

        # 不确定性调制: 将方差映射为保守程度
        self.uncertainty_fc = nn.Linear(1, 1)
        # 先验强度: 控制不确定性的影响力
        self.prior_strength = nn.Parameter(torch.tensor(1.0))

    def _single_forward(self, gate_feat):
        """单次MC采样前向 — dropout在训练和推理都激活"""
        h = F.elu(self.mc_drop_1(self.fc_gate_1(gate_feat)))
        h = F.elu(self.mc_drop_2(self.fc_gate_2(h)))
        logit = self.fc_gate_out(h)
        return logit

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        # 拼接特征
        gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)

        dataflow_checkpoint("est_gate.feat", gate_feat)

        if self.training:
            # 训练时: 单次采样 + KL散度正则
            logit = self._single_forward(gate_feat)
            gate_prob = torch.sigmoid(logit)

            # 额外做一次采样估计方差(轻量)
            with torch.no_grad():
                logit_2 = self._single_forward(gate_feat)
                local_var = (logit - logit_2).pow(2).mean()

            _dbg("est_gate.train_variance",
                 local_var, "decouple")
        else:
            # 推理时: T次MC采样, 取均值+方差
            logit_samples = []
            for _ in range(self.mc_samples):
                # 关键: 推理时也启用dropout做MC采样
                self.mc_drop_1.train()
                self.mc_drop_2.train()
                logit_s = self._single_forward(gate_feat)
                logit_samples.append(logit_s)
            # 恢复eval模式
            self.mc_drop_1.eval()
            self.mc_drop_2.eval()

            logit_stack = torch.stack(logit_samples, dim=0)
            logit = logit_stack.mean(dim=0)
            predictive_var = logit_stack.var(dim=0)

            # 不确定性调制: 方差大 → 门控向0.5收缩
            strength = torch.sigmoid(self.prior_strength)
            shrink = torch.sigmoid(
                self.uncertainty_fc(predictive_var))
            # shrink ∈ (0,1): 1=完全保守(0.5), 0=完全信任
            gate_prob_raw = torch.sigmoid(logit)
            gate_prob = (1 - strength * shrink) * gate_prob_raw \
                + strength * shrink * 0.5

            _dbg("est_gate.mc_var_mean",
                 predictive_var.mean(), "decouple")
            _dbg("est_gate.shrink_mean",
                 shrink.mean(), "decouple")
            _mc_dropout_tracker.record(
                predictive_var.mean().item(),
                gate_prob.mean().item())

        estimation_gate = gate_prob[
            :, -history_data.shape[1]:, :, :]

        _dbg("est_gate.output_gate",
             estimation_gate, "decouple")

        return history_data * estimation_gate
