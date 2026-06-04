from .estimation_gate import EstimationGate
from .residual_decomp import ResidualDecomp
import torch as _th


def diagnose_gate_saturation(gate_module, threshold_lo=0.05, threshold_hi=0.95):
    """检查 EstimationGate 是否饱和.
    如果 sigmoid 输出几乎全在 0 或 1 附近, gate 失去了调节作用.
    返回饱和比例.
    """
    tau = _th.exp(gate_module.log_tau).item()
    # 检查两个 head 的权重范数比
    norm_a = gate_module.head_a_fc1.weight.data.norm().item()
    norm_b = gate_module.head_b_fc1.weight.data.norm().item()
    balance = min(norm_a, norm_b) / max(norm_a, norm_b, 1e-10)
    print(f"[walpurgis:gate] tau={tau:.4f}  head_balance={balance:.4f} "
          f"(1.0=perfect)")
    if balance < 0.3:
        print(f"  ⚠ Heads severely imbalanced — "
              f"one head may be dominating")
    return {"tau": tau, "head_balance": balance}


__all__ = ["EstimationGate", "ResidualDecomp", "diagnose_gate_saturation"]
