import torch
import torch.nn as nn
import sys

_DBG_MASK = ("--dbg-mask" in sys.argv)


class Mask(nn.Module):
    """算法改动: 软阈值 mask
    原版: mask = adj_predefined + 1e-7, 然后做元素乘
    改为: 用 sigmoid 把 predefined adj 做 soft thresholding,
    让边权在 (0,1) 之间连续, 而非 {0, ~1e-7} 的二值
    """
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # 可学习的阈值温度
        self.threshold_temp = nn.Parameter(torch.tensor(5.0))

    def _mask(self, index, adj):
        raw_mask = self.mask[index]
        # 算法改动: soft sigmoid mask
        temp = torch.clamp(self.threshold_temp, min=0.5)
        soft_mask = torch.sigmoid(raw_mask * temp)
        # 保留极小的背景值让梯度能流过
        soft_mask = soft_mask + 1e-7

        result = soft_mask.to(adj.device) * adj

        if _DBG_MASK:
            with torch.no_grad():
                active = (soft_mask > 0.5).float().mean().item()
                print(f"[DBG-MASK] idx={index}  temp={temp.item():.3f}  "
                      f"active_ratio={active:.3f}  "
                      f"result_norm={result.norm().item():.4f}")
        return result

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._mask(index, a))
        return result
