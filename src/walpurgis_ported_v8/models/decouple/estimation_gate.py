import torch
import torch.nn as nn
import sys

_DBG = ("--dbg" in sys.argv)


class EstimationGate(nn.Module):
    """算法改动: dual-path gating
    原版: 把 node_emb + time_emb 拼接 -> FC -> ReLU -> FC -> sigmoid 得到 gate,
          然后 output = gate * history_data
    改为: 两条通路
      path_a: 同样拼接 -> FC -> SiLU -> FC -> sigmoid 得到 gate (SiLU 替代 ReLU)
      path_b: 对 history_data 做一个 1x1 conv (即 pointwise Linear) 得到 bias term
      output = gate * history_data + (1 - gate) * bias_term
    这样 gate=0 时信号不是直接消失, 而是退化到 bias 通路, 梯度更健康
    """

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.SiLU()  # 算法改动: SiLU 替代 ReLU
        self.fc2 = nn.Linear(hidden_dim, 1)
        # bias path: pointwise projection on history data
        self.bias_proj = nn.Linear(1, 1, bias=True)

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        hidden = self.fc1(gate_feat)
        hidden = self.activation(hidden)
        gate = torch.sigmoid(self.fc2(hidden))
        gate = gate[:, -history_data.shape[1]:, :, :]

        # dual-path: gate * x + (1-gate) * bias(x)
        bias_term = self.bias_proj(history_data)
        output = gate * history_data + (1.0 - gate) * bias_term

        if _DBG:
            with torch.no_grad():
                print(f"[DBG][EstimationGate] gate_mean={gate.mean().item():.4f}  "
                      f"gate_std={gate.std().item():.4f}  "
                      f"output_absmax={output.abs().max().item():.4f}", flush=True)
        return output
