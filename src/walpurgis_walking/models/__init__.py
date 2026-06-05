from .trainer import trainer
from .model import D2STGNN

# upstream 用 from models.model import * 通配导入;
# 显式导出 + 模型注册表, 方便扩展多模型对比实验

_MODEL_REGISTRY = {
    "D2STGNN": D2STGNN,
}


def build_model(name, **model_args):
    """按名称从注册表构建模型, 并打印参数统计.
    upstream 直接 D2STGNN(**args), 无注册表无统计.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}', available: {list(_MODEL_REGISTRY)}")
    cls = _MODEL_REGISTRY[name]
    model = cls(**model_args)
    # 参数分类统计: embedding / conv / linear / norm / other
    stats = {"embedding": 0, "conv": 0, "linear": 0, "norm": 0, "other": 0}
    import torch.nn as _nn
    for n, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            bucket = "other"
            if isinstance(m, (_nn.Embedding,)):
                bucket = "embedding"
            elif isinstance(m, (_nn.Conv1d, _nn.Conv2d)):
                bucket = "conv"
            elif isinstance(m, (_nn.Linear,)):
                bucket = "linear"
            elif isinstance(m, (_nn.LayerNorm, _nn.InstanceNorm1d,
                                _nn.InstanceNorm2d, _nn.GroupNorm,
                                _nn.BatchNorm1d, _nn.BatchNorm2d)):
                bucket = "norm"
            stats[bucket] += p.numel()
    total = sum(stats.values())
    print(f"[walpurgis] Model {name}: {total:,} params")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k:12s}: {v:>10,} ({100*v/max(total,1):.1f}%)")
    return model


__all__ = ["trainer", "D2STGNN", "build_model"]
