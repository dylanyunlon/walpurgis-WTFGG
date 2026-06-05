from .dataloader import DataLoader
import numpy as _np

# upstream: from dataloader.dataloader import DataLoader, 一行结束.
# walpurgis改动: 加 validate_loader 对加载后的数据做完整性检查.


def validate_loader(loader, name=""):
    """校验 DataLoader 内部数据的完整性.
    检查: nan 比例 / inf 比例 / 常数列(方差=0) / batch 尺寸一致性.
    upstream 无此类校验, 数据问题要跑到 loss 计算才暴露.
    """
    xs, ys = loader.xs, loader.ys
    issues = []

    for tag, arr in [("x", xs), ("y", ys)]:
        nan_frac = _np.isnan(arr).mean()
        inf_frac = _np.isinf(arr).mean()
        if nan_frac > 0:
            issues.append(f"{tag} has {nan_frac:.2%} NaN values")
        if inf_frac > 0:
            issues.append(f"{tag} has {inf_frac:.2%} Inf values")
        # 检查常数列(最后一维的每个 feature 方差)
        feat_std = _np.nanstd(arr.reshape(-1, arr.shape[-1]), axis=0)
        const_feats = _np.where(feat_std < 1e-10)[0]
        if len(const_feats) > 0:
            issues.append(f"{tag} has {len(const_feats)} constant features "
                          f"at indices {const_feats.tolist()}")

    # batch 尺寸
    last_bs = loader.size % loader.batch_size
    if last_bs != 0 and last_bs != loader.batch_size:
        issues.append(f"last batch has {last_bs} samples "
                      f"(expected {loader.batch_size})")

    prefix = f"[{name}] " if name else ""
    if issues:
        print(f"[walpurgis:validate] {prefix}{len(issues)} issues found:")
        for iss in issues:
            print(f"  ⚠ {iss}")
    else:
        print(f"[walpurgis:validate] {prefix}OK — "
              f"x={xs.shape}, y={ys.shape}, "
              f"batches={loader.num_batch}")
    return issues


__all__ = ["DataLoader", "validate_loader"]
