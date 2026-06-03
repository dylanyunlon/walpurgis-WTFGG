"""
walpurgis_ported_v10 — 第二位 Claude (M026-M050) 鲁迅式移植
==========================================================
Upstream: D2STGNN (upstream/d2stgnn)
算法改动策略 ≥20%:
  - 损失: smooth Huber + log-cosh 混合, 可切换 quantile loss (τ=0.5)
  - 估计门: Swish 激活, 双头注意力加权(不同 W_q W_k), GroupNorm
  - 残差分解: Mish 激活, 可学习残差缩放系数 α (init 0.9)
  - 时空卷积: InstanceNorm 替代 BN, gconv 内加 skip connection
  - 扩散预测: Cosine 退火 AR dropout, 线性插值 padding
  - 扩散块: 3-layer MLP backcast + GELU, residual gating
  - 距离函数: 多头(3-head) Q-K 注意力, InstanceNorm, Dropout 正则
  - 图掩码: softplus 阈值 + 温度衰减 τ_anneal, 对角清零
  - 归一化: 双向对称 D^{-1/2}AD^{-1/2} + 高阶指数衰减 λ^k (0.8)
  - 动态图: 可学习时间卷积权重 (softmax), cosine-sim 辅助
  - GRU: 步间 RMSNorm + gradient checkpoint
  - Transformer: Rotary PE, flash-attn-style mask, 注意力熵监控
  - 固有预测: 可学习步长衰减 exp(-γ·step)
  - 固有块: 残差门控 (sigmoid gate), gradient checkpoint
  - 主模型: Mish 输出激活, 层权重 softmax 聚合 + 温度
  - 训练器: 自适应 p90 梯度裁剪, warmup-cosine 学习率
  - 数据: 周期性 sin/cos 编码, Tukey fences 异常剔除
  - 邻接: RBF kernel + k-NN(15) 稀疏化 + 双向对称闭包
  - DataLoader: 环形 wrap padding, Knuth shuffle, 3-tuple yield

调试系统:
  设置 WALPURGIS_V10_DEBUG=1  开启所有调试打印
  设置 WALPURGIS_V10_DEBUG=model,trainer  只开启指定模块
"""

import os as _os

_DEBUG_ENV = _os.environ.get("WALPURGIS_V10_DEBUG", "")
_DEBUG_ALL = (_DEBUG_ENV == "1")
_DEBUG_TAGS = set(_DEBUG_ENV.split(",")) if _DEBUG_ENV and not _DEBUG_ALL else set()


def _dbg(tag: str, msg: str, **tensors):
    """统一调试打印入口.

    使用方法:
        from walpurgis_ported_v10 import _dbg
        _dbg("model", "forward pass", x=some_tensor, adj=adj_tensor)

    打印示例:
        [v10:model] forward pass | x: shape=(32,12,207,32) dtype=float32 min=-2.31 max=5.12 mean=0.03 nan=0 inf=0
    """
    if not (_DEBUG_ALL or tag in _DEBUG_TAGS):
        return
    parts = [f"[v10:{tag}] {msg}"]
    for name, t in tensors.items():
        import torch as _th
        if isinstance(t, _th.Tensor):
            s = (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
                 f"min={t.min().item():.4g} max={t.max().item():.4g} "
                 f"mean={t.float().mean().item():.4g} "
                 f"nan={t.isnan().sum().item()} inf={t.isinf().sum().item()}")
        elif isinstance(t, (list, tuple)):
            s = f"{name}: len={len(t)} types={[type(x).__name__ for x in t[:3]]}"
        else:
            s = f"{name}: {type(t).__name__}={t}"
        parts.append(s)
    print(" | ".join(parts), flush=True)
