import torch as _th
from .mask import Mask
from .normalizer import Normalizer, MultiOrder
from .distance import DistanceFunction


def validate_graph_pipeline(dist_fn, mask, normalizer, multi_order,
                            sample_X, E_d, E_u, T_D, D_W):
    """端到端校验: 跑一遍 dist→mask→norm→multi_order 管线,
    打印每步输出的形状/稀疏度/数值范围.

    upstream 无此类管线诊断; 这里让你在实验前确认
    各组件衔接正常、没有 nan/inf 泄漏.

    用法:
        from models.dynamic_graph_conv.utils import validate_graph_pipeline
        validate_graph_pipeline(
            model.dynamic_graph_constructor.distance_function,
            model.dynamic_graph_constructor.mask,
            model.dynamic_graph_constructor.normalizer,
            model.dynamic_graph_constructor.multi_order,
            dummy_X, model.node_emb_d, model.node_emb_u, dummy_tid, dummy_diw)
    """
    print(f"{'=' * 64}")
    print("[walpurgis] Graph pipeline validation")

    with _th.no_grad():
        adj_list = dist_fn(sample_X, E_d, E_u, T_D, D_W)
        for i, a in enumerate(adj_list):
            nnz = (a.abs() > 1e-7).float().mean().item()
            print(f"  [dist]  adj[{i}] shape={tuple(a.shape)} "
                  f"sparsity={1-nnz:.2%} range=[{a.min():.4f}, {a.max():.4f}] "
                  f"nan={a.isnan().any().item()}")

        masked = mask(adj_list)
        for i, a in enumerate(masked):
            nnz = (a.abs() > 1e-7).float().mean().item()
            print(f"  [mask]  adj[{i}] sparsity={1-nnz:.2%} "
                  f"range=[{a.min():.4f}, {a.max():.4f}]")

        normed = normalizer(masked)
        for i, a in enumerate(normed):
            row_sum = a.sum(dim=-1)
            print(f"  [norm]  adj[{i}] row_sum μ={row_sum.mean():.4f} "
                  f"σ={row_sum.std():.4f} "
                  f"nan={a.isnan().any().item()} inf={a.isinf().any().item()}")

        multi = multi_order(normed)
        for i, order_list in enumerate(multi):
            for k, g in enumerate(order_list):
                decay = g.abs().mean().item()
                print(f"  [multi] modality={i} order={k+1} "
                      f"mean_weight={decay:.6f}")

    print(f"{'=' * 64}")


__all__ = ["Mask", "Normalizer", "MultiOrder", "DistanceFunction",
           "validate_graph_pipeline"]
