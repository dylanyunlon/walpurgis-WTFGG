from .inh_block import InhBlock
import torch as _th


def build_inh_block(hidden_dim, num_heads=4, forecast_hidden_dim=256,
                    **model_args):
    """工厂函数: 构建 InhBlock 并做参数完整性断言.

    upstream 直接 InhBlock(...) 无任何校验;
    v10 在构建后立即检查关键子模块的维度一致性,
    防止 hidden_dim / forecast_dim 不匹配导致运行时 shape error.
    """
    blk = InhBlock(hidden_dim, num_heads=num_heads,
                   forecast_hidden_dim=forecast_hidden_dim, **model_args)
    # 校验 backcast MLP 输入输出维度一致
    mlp_in = blk.backcast_mlp[0].in_features
    mlp_out = blk.backcast_mlp[-1].out_features
    assert mlp_in == mlp_out == hidden_dim, (
        f"backcast MLP 维度不一致: in={mlp_in}, out={mlp_out}, "
        f"expected={hidden_dim}")
    # 校验 res_gate_fc 维度
    assert blk.res_gate_fc.in_features == hidden_dim
    return blk


def trace_inh_activations(blk, sample_input):
    """诊断工具: 给一个 (B,L,N,D) 张量, 跑一遍 forward 并打印每个中间激活的统计.

    用法:
        from models.inherent_block import trace_inh_activations
        trace_inh_activations(model.layers[0].inh_layer, dummy_x)
    """
    blk.eval()
    hooks = []
    activation_log = {}

    def _make_hook(name):
        def _hook(mod, inp, out):
            t = out if isinstance(out, _th.Tensor) else out[0]
            activation_log[name] = {
                "shape": tuple(t.shape),
                "mean": t.float().mean().item(),
                "std": t.float().std().item(),
                "abs_max": t.float().abs().max().item(),
                "has_nan": bool(t.isnan().any()),
            }
        return _hook

    for n, m in blk.named_modules():
        if n:
            hooks.append(m.register_forward_hook(_make_hook(n)))

    with _th.no_grad():
        blk(sample_input)

    for h in hooks:
        h.remove()

    print(f"{'─' * 72}")
    print(f"InhBlock activation trace  input={tuple(sample_input.shape)}")
    for name, stats in activation_log.items():
        flag = " ⚠NaN!" if stats["has_nan"] else ""
        print(f"  {name:40s} {str(stats['shape']):20s} "
              f"μ={stats['mean']:+.4f} σ={stats['std']:.4f} "
              f"|max|={stats['abs_max']:.4f}{flag}")
    print(f"{'─' * 72}")
    return activation_log


__all__ = ["InhBlock", "build_inh_block", "trace_inh_activations"]
