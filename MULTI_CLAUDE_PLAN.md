# Walpurgis-WTFGG 多Claude协作开发计划
# =========================================
# 总体目标: 对upstream D2STGNN代码进行大规模算法改写(~20%),
# 每个walpurgis变体都包含独立的算法修改。
# 规则: 直接push到main分支, 不开新分支, 不加v2/v3/port后缀
# Git author: dylanyunlon <dogechat@163.com>
# =========================================

## 里程碑分配

### 第一位Claude (当前，已完成): M001-M003
- M001: ✅ 创建 walpurgis_zenith 完整变体 (37个文件)
  - SpectralDecayGate, LayerAttentionAggregator
  - GELU输出头, OneCycleLR, temporal_smoothness
  - 完整诊断工具链
- M002: ✅ SYNTH数据集smoke test通过 (MAE 20→5.7)
- M003: ✅ Push到main, 调度子Claude任务

### 第二位Claude (Opus 4.6): M004-M007
- M004: 拉取 dylanyunlon/claude-hk-config 同步cookie
- M005: 对 walpurgis_aurora 变体进行算法改写
  - 改动方向: 注意力机制重新设计
  - Multi-Scale Temporal Attention (替代单一GRU)
  - Gated Fusion替代简单的残差加法
  - 自适应图正则化 (spectral graph regularizer)
- M006: aurora变体SYNTH test通过
- M007: Push到main

### 第三位Claude (Opus 4.6): M008-M011
- M008: 拉取最新main + claude-hk-config
- M009: 对 walpurgis_eclipse 变体进行算法改写
  - 改动方向: 时间序列分解增强
  - Learnable Wavelet Decomposition (替代固定FFT)
  - Cross-Attention between diffusion/inherent branches
  - Gradient-aware loss weighting
- M010: eclipse变体SYNTH test通过
- M011: Push到main

### 第四位Claude (Opus 4.6): M012-M015
- M012: 拉取最新main + claude-hk-config
- M013: 对 walpurgis_equinox 变体进行算法改写
  - 改动方向: 图结构学习
  - Differentiable Graph Sampling (Gumbel-Softmax)
  - Node-adaptive positional encoding
  - Contrastive regularization on graph embeddings
- M014: equinox变体SYNTH test通过
- M015: Push到main

### 第五位Claude (Opus 4.6): M016-M019
- M016: 拉取最新main + claude-hk-config
- M017: 对 walpurgis_solstice 变体进行算法改写
  - 改动方向: 损失函数和训练策略
  - Multi-horizon adaptive loss (per-step weighting)
  - Stochastic Weight Averaging (SWA) 
  - Progressive training (easy→hard sample curriculum)
- M018: solstice变体SYNTH test通过
- M019: Push到main

### 第六位Claude (Opus 4.6): M020-M023
- M020: 拉取最新main + claude-hk-config
- M021: 对 walpurgis_nebula 变体进行算法改写
  - 改动方向: 模型架构创新
  - Mixture-of-Experts routing in DecoupleLayer
  - Reversible Residual connections (内存优化)
  - Learned input augmentation (noise injection schedule)
- M022: nebula变体SYNTH test通过
- M023: Push到main, 整体集成测试

## 关键规则 (所有子Claude必须遵守):
1. 改的是算法，不是字符串/docstring/str_replace
2. 直接push到main，不开新分支，不加v2/v3/port后缀
3. git author: dylanyunlon <dogechat@163.com>
4. GitHub token: <GH_TOKEN_FROM_ENV>
5. 先拉取 dylanyunlon/claude-hk-config 同步cookie
6. 每个变体必须通过SYNTH smoke test才能push
7. 参考 walpurgis_zenith 的文件结构和诊断工具模式
