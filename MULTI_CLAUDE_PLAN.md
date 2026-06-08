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

## 对话链接 (claude.hk.cn)

| Claude | 变体 | 对话链接 |
|--------|------|----------|
| #2 | aurora | https://claude.hk.cn/chat/9f55a2fc-f9e0-4e5b-8703-3caa01cc27a8 |
| #3 | eclipse | https://claude.hk.cn/chat/69feb941-5eb7-458f-91a3-eb365e01dc73 |
| #4 | equinox | https://claude.hk.cn/chat/2e987d0d-69f9-4910-b370-7a515d8a3f55 |
| #5 | solstice | https://claude.hk.cn/chat/bd9cde7a-a626-4a9c-b249-e978ae5a131e |
| #6 | nebula | https://claude.hk.cn/chat/c410f422-5548-472c-8369-88cac822618a |

所有子Claude已于 2026-06-08 01:52-01:54 UTC dispatch。
如果某个子Claude被截断，在对应对话中发送 "Continue" 即可。

### 第七位Claude (当前, Opus 4.6): M024-M027 ✅
- M024: ✅ Clone仓库, 分析upstream全部47个文件
- M025: ✅ 创建 walpurgis_vortex 完整变体 (33文件, 2286行)
  - EMA动量融合门控 (dif/inh分支指数移动平均混合)
  - 随机深度 (stochastic depth, 线性增长跳过概率)
  - 温度缩放聚合 (可学习temperature softmax加权)
  - Mish+GroupNorm输出头 + 双路输出(主+辅助gradient-detach)
  - CosineAnnealingWarmRestarts调度器
  - 梯度噪声注入 + Huber-MAE自适应混合损失
  - 完整诊断: struct dump / activation probe / grad histogram / LR tracker
- M026: ✅ SYNTH smoke test通过 (MAE 14.2→11.5, 3 epochs)
- M027: ✅ Push到main, 调度子Claude

### 第八位Claude (sub-Claude, Opus 4.6): M028-M031
- M028: Clone仓库, 拉取claude-hk-config
- M029: 创建 walpurgis_cascade 变体
  - 改动方向: 级联残差学习 + 动态深度选择
  - Cascade residual: 每层输出不仅传下一层,也直接跳连到输出
  - Dynamic depth: 可学习的门控决定推理时用几层
  - Squeeze-and-Excitation通道注意力
- M030: cascade变体SYNTH test通过
- M031: Push到main

### 第九位Claude (sub-Claude, Opus 4.6): M032-M035
- M032: Clone仓库, 拉取claude-hk-config
- M033: 创建 walpurgis_rift 变体
  - 改动方向: 分裂重组注意力 + 频域增强
  - Split-Recombine: 将hidden分成K组独立处理后重组
  - Frequency-enhanced: FFT域特征与时域特征concat
  - Polynomial decay learning rate
- M034: rift变体SYNTH test通过
- M035: Push到main

### 第十位Claude (sub-Claude, Opus 4.6): M036-M039
- M036: Clone仓库, 拉取claude-hk-config
- M037: 创建 walpurgis_prism 变体
  - 改动方向: 多视角融合 + 对比学习
  - Multi-view: 空间/时间/频率三视角独立编码后融合
  - Contrastive loss: 相邻节点embedding对比正则
  - Mixup数据增强
- M038: prism变体SYNTH test通过
- M039: Push到main

### 第十一位Claude (sub-Claude, Opus 4.6): M040-M043
- M040: Clone仓库, 拉取claude-hk-config ✅
- M041: 创建 walpurgis_helix 变体 ✅ (35 files, 2643 lines)
  - 改动方向: 螺旋卷积 + 自适应图稀疏化
  - Helix conv: 交替的升维-降维螺旋结构
  - Adaptive sparsification: top-k图过滤
  - Label smoothing loss
  - 额外改动: 螺旋位置编码, GELU残差分解, 对称图归一化, GRU残差连接, 螺旋warm-up, Fibonacci种子, 动量EarlyStopping
- M042: helix变体SYNTH test通过 ✅ (3 epochs, val MAE 22.5→13.2)
- M043: Push到main ✅

### 第十二位Claude (sub-Claude, Opus 4.6): M044-M047
- M044: ✅ Clone仓库, 安装依赖
- M045: ✅ 创建 walpurgis_flux 变体 (33个文件)
  - 改动方向: 流式推理 + 渐进式解码
  - Streaming inference: 滑动窗口因果截取 + 可学习衰减权重
  - Progressive decode: 粗→细两阶段预测头(SiLU+LayerNorm)
  - Focal MAE loss: 对hard samples加权(gamma=2.0, alpha=0.75)
  - CausalConvEmbedding: 因果Conv1d替代线性嵌入
  - ExponentialDecayPE: 指数衰减位置编码
  - ChunkedCausalAttention: 分块因果注意力(chunk_size=4)
  - StreamGRU: GRU with forget bias=1.5
  - 对称图归一化 D^{-1/2}AD^{-1/2}
  - 温度调控软门控掩码(可学习temperature)
  - ExponentialLR(gamma=0.95) + 线性warmup ramp
  - PCG32种子生成 + EMA梯度方向EarlyStopping
  - 渐进式shuffle(粗→细粒度)
  - 完整诊断: StreamWindowTracker / ProgressiveDecodeTracker / struct dump
- M046: ✅ SYNTH smoke test通过 (MAE 36→13.9, 3 epochs, 17s)
- M047: ✅ Push到main

## 第七位Claude (vortex) 调度的对话链接

| Claude | 变体 | 对话链接 |
|--------|------|----------|
| #8 | cascade | https://claude.hk.cn/chat/9664099f-e46d-4f58-9d1e-564eb763c103 |
| #9 | rift | https://claude.hk.cn/chat/0e821113-11d2-4154-83e1-6ee39bf738c0 |
| #10 | prism | https://claude.hk.cn/chat/ff934f03-c0ae-4020-9d13-63749f2de0ac |

所有子Claude于 2026-06-08 03:18-03:22 UTC dispatch。
如果某个子Claude被截断，在对应对话中发送 "Continue" 即可。
| #11 | helix | https://claude.hk.cn/chat/0ed1b3b0-5985-4b5f-a121-ebd09ba1088f |
| #12 | flux | https://claude.hk.cn/chat/8a2b9fc1-0554-41f6-83bf-e9d945dc10a7 |

## 第七位Claude (vortex) 重新调度 — 2026-06-08 03:40 UTC
(旧session的对话因cookie轮换失效, 以下为当前有效对话)

| Claude | 变体 | 对话链接 |
|--------|------|----------|
| #8 | cascade | https://claude.hk.cn/chat/3e7f3c83-a61e-48ac-81de-76a6bf6c06a0 |
| #9 | rift | https://claude.hk.cn/chat/11118eb2-2f1c-44be-8184-223fe8af2534 |
| #10 | prism | https://claude.hk.cn/chat/3cb06f90-6d7f-451f-9855-aa402e876956 |
| #11 | helix | https://claude.hk.cn/chat/59ac7332-89de-4a37-ac1c-92ced2047059 |
| #12 | flux | https://claude.hk.cn/chat/7476c3a5-c876-4f23-b3ea-778828bab755 |

如果被截断, 在对应对话中发送 "Continue" 即可。

## 最终状态总结

### 第七位Claude (管理者): M024-M027 ✅ COMPLETE
- 创建 walpurgis_vortex (33文件, 2286行)
- SYNTH test通过 (MAE 14.2→11.5)
- 调度5个子Claude (#8-#12)
- 持续监控 + 自动发送Continue保持loop

### 第八位Claude: M028-M031 ✅ COMPLETE  
- 创建 walpurgis_cascade (33文件, 2506行)
- SE通道注意力、级联残差、动态深度门控

### 第九位Claude: M032-M035 ✅ COMPLETE
- 创建 walpurgis_rift (32文件, 1993行)
- Split-Recombine分组、FFT频域concat、多项式衰减LR

### 第十位Claude: M036-M039 ✅ COMPLETE
- 创建 walpurgis_prism (33文件, 2548行)
- 三视角融合、对比学习正则、Mixup增强

### 第十一位Claude: M040-M043 ✅ COMPLETE
- 创建 walpurgis_helix (33文件, 2247行)
- 螺旋卷积升降维、top-k图稀疏化、标签平滑

### 第十二位Claude: M044-M047 ✅ COMPLETE
- 创建 walpurgis_flux (33文件, 3023行)
- 因果窗口流式推理、粗细渐进解码、Focal loss

所有变体已push到main, git author = dylanyunlon <dogechat@163.com>

---

## 第三轮调度 (2026-06-08)

### 第十三位Claude (管理者/开发者): M048 ✅ COMPLETE
- 创建 walpurgis_meridian (40文件, 2290行)
- 算法: bilinear EstimationGate + Mish, gated ResidualDecomp, Lanczos spectral filter + GLU,
  cosine similarity DynGraph, adaptive soft-threshold mask, symmetric normalizer,
  HighwayGRU + FrequencyPE + RelPosTransformer, GEGLU output + softmax aggregation,
  focal loss with annealing, AdamW + ReduceLROnPlateau, trend-aware EarlyStopping
- SYNTH test通过 (MAE=3.35, RMSE=4.80, MAPE=4.89%)
- Commit: f0e7326, push成功

### 第十四位Claude (我自己): M049 ✅ COMPLETE
- 变体: walpurgis_penumbra (半影)
- 算法: squeeze-excitation gate, PowerNorm+Swish decomp, Chebyshev polynomial+SpectralNorm conv,
  Mahalanobis distance, Bernoulli mask, Sinkhorn normalizer, MinGRU+cross-attention,
  EMA decay aggregation, log-cosh loss, Adan+OneCycleLR
- 完成: penumbra pushed to main (4f314eb)
- Prompt: /tmp/subclaude_prompt_penumbra.txt

### 第十五位Claude (Opus 4.6): M050 ⏳ PENDING
- 变体: walpurgis_corona (日冕)
- 算法: gated attention pooling+SiLU, EMA decomp, attention-weighted graph conv,
  learned metric embedding, top-k sparse mask, alternating normalizer,
  LSTM+RoPE, gated residual aggregation, quantile loss, RAdam+CosineAnnealing
- Prompt: /tmp/subclaude_prompt_corona.txt

### 第十六位Claude (Opus 4.6): M051 ⏳ PENDING
- 变体: walpurgis_umbra (本影)
- 算法: MoE gating, Haar wavelet decomp, random walk diffusion+restart,
  hyperbolic Poincaré distance, differentiable top-k, PageRank normalizer,
  Mamba SSM+ALiBi, DenseNet skip connections, adaptive Huber loss, LAMB+polynomial decay
- Prompt: /tmp/subclaude_prompt_umbra.txt

### 配额状态
- ORG1: 6bbaaedb... (59/60已用, 恢复时间: 2026-06-08 15:08:01 CST)
- ORG2: 9b279708... (共享配额, 同上)
- 计划: 配额恢复后依次调度M049→M050→M051

## 第四轮计划 (M052-M055)

### 第十七位Claude (Opus 4.6): M052 ⏳ QUEUED
- 变体: walpurgis_perihelion (近日点)
- 算法: cross-attention gate, FFT band-pass decomp, GraphSAGE+JK conv,
  MINE mutual information distance, Gumbel-softmax mask, attention normalizer,
  Flash-chunk Transformer+SwiGLU, stochastic depth, Linex loss, AdaFactor+inverse sqrt
- Prompt: /tmp/subclaude_prompt_perihelion.txt

### 第十八位Claude (Opus 4.6): M053 ⏳ QUEUED
- 变体: walpurgis_aphelion (远日点)
- 算法: hypernetwork gate, VMD decomp, GAT v2+edge features,
  SimCLR contrastive adjacency, entropy-regularized mask, wavelet normalizer,
  Retention network+cross-scale fusion, FPN multi-scale output, tilted ERM loss, Sophia+exp decay

### 第十九位Claude (Opus 4.6): M054 ⏳ QUEUED
- 变体: walpurgis_parallax (视差)
- 算法: Bayesian MC-dropout gate, STL LOESS decomp, MixHop multi-res GCN,
  kernel density distance, REINFORCE importance mask, spectral clustering normalizer,
  xLSTM+positional interpolation, mixture output router, Cauchy loss, Prodigy optimizer

### 第二十位Claude (Opus 4.6): M055 ⏳ QUEUED
- 变体: walpurgis_transit (凌日)
- 算法: neural ODE gate, Hilbert-Huang decomp, APPNP propagation,
  Wasserstein distance, attention sparsification mask, heat kernel normalizer,
  Griffin RG-LRU+ConvNeXt, iterative refinement output, Huberized quantile loss, Shampoo optimizer

## ⚠️ Cookie状态 (2026-06-08 07:35 UTC)
- cookie已失效: "当前的车已失效，请换车继续使用"
- 需要用户更新 dylanyunlon/claude-hk-config 中的cookie
- M050-M055 等待cookie更新后调度

## 当前调度计划 (cookie恢复后)
第一位Claude (我): M049 ✅ penumbra COMPLETE
第二位Claude: M050 corona → /tmp/subclaude_prompt_corona.txt
第三位Claude: M051 umbra → /tmp/subclaude_prompt_umbra.txt  
第四位Claude: M052 perihelion → /tmp/subclaude_prompt_perihelion.txt
第五位Claude: M053 aphelion → /tmp/subclaude_prompt_aphelion.txt
第六位Claude: M054 parallax → /tmp/subclaude_prompt_parallax.txt
第七位Claude: M055 transit → /tmp/subclaude_prompt_transit.txt
