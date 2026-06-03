# Walpurgis-WTFGG: Multi-Claude 开发进度总览

> 每位 Claude 接力完成一段里程碑区间，形成连续的 M001→M??? 编号链。
> 下一位 Claude 开新对话时，把此文件 + 最新 git log 交给它即可无缝衔接。

---

## 已完成的 Claude 接力链

### 第一位 Claude — M001-M014: 底层异构内存基础设施 (C++/CUDA)
```
M001-M004  core tiered allocator, temporal bridge, migration scheduler
M005-M006  review — lockfree touch, shared_mutex, binary search
M007       seqlock for wait-free partition reads + adaptive partitioning
M008       per-tier slab allocator with bitmask pages
M009       TierPtr RAII guard + AsyncMigrationEngine
M010       async migration benchmark + publication data
M011-M012  IntervalIndex + indexed temporal queries + benchmark
M013-M014  augmented interval skip list for O(log P) partition selection
```
**产出**: `src/core/`, `src/bridge/`, `src/scheduler/`, `src/bench/`, `src/cuda/`

### 第二位 Claude — M001-M075: 论文写作 (LaTeX)
```
M001-M025  Section 1 (Introduction) + Section 2 (System/Algorithm) — Philemon-TSH
M026-M050  rename Philemon→Walpurgis + Section 3 (Correctness & Performance)
M051-M075  Section 4 (Experimental Design) — Walpurgis
```
**产出**: `walpurgis_reconstructed.tex`, `template_extraction_walpurgis_EN.tex`

### 第三位 Claude — M101-M200: D2STGNN 移植与首次算法改写 (Python/PyTorch)
```
M101-M125  full-paper walpurgis_reconstructed.tex (modeled on des_loc_reconstructed.tex)
M126-M150  port D2STGNN into src/walpurgis/ with tier-aware instrumentation
M151-M175  deep instrumentation for remaining under-ported D2STGNN files
M176-M200  algorithmic divergence + deep debug instrumentation (v1 rewrite)
```
**产出**: `src/walpurgis/` 全部文件初版, `src/walpurgis_ported/` (备份)

### 第四位 Claude — M201-M255: 第二次全量改写 v1→v2 (Python/PyTorch)
```
M201-M205  v2 core model — TensorProbe, Charbonnier loss, adaptive gradient clipping
M206-M210  v2 main + utils — EpochProfiler, gradient watchdog, SHA-256 checkpoints
M211-M215  v2 dynamic graph conv — cosine distance, soft-threshold mask, symmetric norm
M216-M220  v2 diffusion block + decouple — support decay, tanh gate, learnable residual
M221-M225  v2 inherent block + dataloader — GRU, pre-norm Transformer, learnable PE
M226-M230  v2 core model — softmax-logit aggregation, Gumbel-sigmoid gate, Huber-Charbonnier
M231-M235  v2 trainer — percentile IQR clipping, cosine CL ramp, relative gradient forensics
M236-M240  v2 dynamic graph — EMA density rescue, cosine cache, degree-scalable norm
M241-M245  v2 diffusion + decouple — attention-weighted conv, softplus gate, scale+shift decomp
M246-M250  v2 inherent block — BiGRU, RoPE attention, stochastic depth, gradient checkpoint
M251-M255  v2 utils + main — CRC32 checkpoints, plateau detector, exp-backoff watchdog
```
**产出**: 全部19个Python文件 v2版, 4个YAML配置 v2版

### 第五位 Claude — M256-M274: 第三次全量改写 v2→v3
```
M256-M262  v3 leaf modules (residual_decomp, estimation_gate, normalizer, mask, distance, forecasts)
M263-M267  v3 block modules (inh_model, inh_block, dif_block, dif_model, losses)
M268-M270  v3 core (model, trainer, dy_graph_conv)
M271-M274  v3 infrastructure (main, train, log, dataloader, configs)
```
**产出**: 19个Python文件 v3版 (6644行), 4个YAML v3版

### 第六位 Claude — M275-M299: 第四次全量改写 v3→v4
```
M275-M281  v4 leaf modules (SwiGLU gate, Mish gate, spectral norm, hysteresis mask, etc.)
M282-M286  v4 block modules (RoPE-lite, cosine-cyclic drop, RMSNorm, stochastic routing, etc.)
M287-M289  v4 core (MoE gating, inverse-sqrt schedule, Gumbel-softmax, spectral SVD hash)
M290-M294  v4 infrastructure (adaptive backoff, spectral gradient co-tracking, JSONL logging)
M295-M299  v4 data generators + validation
```
**产出**: 42个文件 v4版 (7632行), `src/walpurgis_ported/`

### 第七位 Claude — M300-M324: 鲁迅式 v2 移植 upstream→walpurgis_ported_v2
```
M300-M324  见上一版本进度记录（略）
```
**产出**: `src/walpurgis_ported_v2/` — 41 .py + 4 .yaml, 2615行 Python

### 第八位 Claude — M325-M349: 鲁迅式 v3 移植 upstream→walpurgis_ported_v3 ✅
```
M325-M349  完整upstream port, ~20%算法变形 + debug开关
```
**产出**: `src/walpurgis_ported_v3/` — 41 .py + 4 .yaml, 2202行 Python

### 第九位 Claude — M350-M374: git-am patch + relay plan ✅
```
M350-M374  patch生成 + relay plan文档
```

### 第十位 Claude — M375-M399: v4 鲁迅式移植 ✅
```
M375-M399  walpurgis_ported_v4 完整port
```

### 第十一位 Claude — M400-M424: v5 鲁迅式移植 ✅
```
M400-M424  walpurgis_ported_v5 — D2STGNN port with ~20% algorithmic deltas
```

### 第十二位 Claude (当前) — M425-M449: v6 鲁迅式移植 ✅ 已完成
```
M425-M427  v6 debug system (__init__.py) + configs (gap 3→4, lrate→0.0018)
M428-M429  v6 dataloader (ring buffer, 3-tuple yield) + datasets (CSR sparse analysis)
M430-M433  v6 utils — cal_adj (eps-clamp, shift-invert ARPACK, isolated-node guard),
           load_data (Welford scaler, integrity checkpoint),
           log (JSON Lines, gradient norm hooks),
           train (EMA early stopping, shape assertion)
M434-M436  v6 losses (Huber-like soft-clip MAE, MAPE floor clamp),
           estimation_gate (GELU + learnable temperature τ),
           residual_decomp (learnable scale, pre-LN option)
M437-M439  v6 diffusion block — dif_model (gconv residual skip, GroupNorm),
           forecast (adaptive gap stride),
           dif_block (feature-attention gate on backcast)
M440-M442  v6 dynamic graph — distance (dual-head attention, TS residual, LN),
           mask (soft-threshold sigmoid, learnable α),
           normalizer (symmetric D^-1/2 A D^-1/2, matrix_power)
M443-M445  v6 inherent block — inh_model (GRU inter-step LN, pre-norm Transformer,
           attention entropy diagnostic),
           forecast (learnable step-decay γ^step),
           inh_block (RoPE replaces additive sinusoidal PE)
M446-M447  v6 core — model (highway embedding gate, output residual shortcut),
           trainer (adaptive p95 grad clip, cosine CL warmup, per-horizon weighted metrics)
M448-M449  v6 main (configurable seed, model hash, output health check) +
           data generators (cyclic sin/cos encoding, variance-aware split,
           Gaussian kernel adj, k-NN sparsification, robust percentile MinMax)
```
**产出**: `src/walpurgis_ported_v6/` — 42 files (38 .py + 4 .yaml), 3276行 Python
**改写策略**: upstream骨架 + ≥20%实质算法改动(非字符串/注释替换) + 全局_dbg()断点系统
**核心算法改动清单**:
  - 损失函数: Huber-like软截断(delta=10), MAPE floor clamp(1e-4)
  - 估计门: ReLU→GELU, 可学习温度τ控制sigmoid锐度
  - 残差分解: 可学习scale因子, pre-LN/post-LN可选
  - 时空卷积: gconv加残差skip, BatchNorm→GroupNorm(4), fill_diagonal_
  - 扩散预测: 自适应gap步长(non-divisible seq_len)
  - 扩散块: backcast后加feature-attention gate
  - 距离函数: 双头(2-head)注意力, TS特征残差shortcut, BN→LN
  - 图掩码: 硬二值→可学习α的soft-threshold sigmoid
  - 归一化: 对称模式(D^-1/2 A D^-1/2), matrix_power替代循环matmul
  - GRU: 步间LayerNorm防隐状态漂移
  - Transformer: pre-norm架构, 注意力熵诊断
  - 固有预测: 可学习步衰减γ^step
  - 位置编码: RoPE旋转位置编码替代加性正弦PE
  - 主模型: highway embedding gate, 输出残差shortcut
  - 训练器: 自适应梯度裁剪(p95滚动窗口), 余弦CL warmup
  - 数据生成: cyclic sin/cos时间编码, 方差感知分层split
  - 邻接矩阵: 高斯核加权, k-NN稀疏化+对称闭合
  - 归一化: 鲁棒百分位MinMax(PEMS04/08)
  - 标准化器: Welford在线算法
  - Laplacian: shift-invert ARPACK, 孤立节点self-loop注入
  - 日志: JSON Lines, 逐层梯度范数追踪hook
  - 早停: EMA平滑(α=0.3)判停
  - DataLoader: 3-tuple yield(x, y, meta)

### 第十三位 Claude — M450-M474: v9 鲁迅式移植 ✅ 已完成
```
M450-M452  v9 debug system (__init__.py, WALPURGIS_V9_DEBUG=1) + configs (verbatim copy)
M453-M454  v9 dataloader (circular wrap padding, 3-tuple yield with sample indices,
           Fisher-Yates in-place shuffle)
M455-M458  v9 utils — cal_adj (nan_to_num single-call, ε=1e-8 isolated node regularisation,
           optional self-loop injection in transition_matrix),
           load_data (Welford online StandardScaler, dispatch-dict load_adj,
           new 'spectral' adj type via shift-invert ARPACK low-pass reconstruction,
           re_max_min clamp(0,1)),
           log (JSON Lines structured logging, per-layer gradient norm backward hooks),
           train (EMA smoothing α=0.25 for EarlyStopping, SHA-256 model fingerprint)
M459-M461  v9 losses (Huber-like soft-clip MAE δ=10 with log-compression beyond δ,
           MAPE floor clamp 1e-5, NEW masked_quantile_loss pinball loss,
           metric() returns 4-tuple),
           estimation_gate (ReLU→GELU, learnable temperature τ, pre-output LayerNorm),
           residual_decomp (ReLU→SiLU, learnable scale γ, pre_ln/post_ln option)
M462-M464  v9 diffusion block — dif_model (BN→GroupNorm(4), gconv residual skip +X_0,
           second FC with Mish activation, fill_diagonal_ in get_graph),
           forecast (AR inter-step Dropout(0.1), ceil gap, pre-FC LayerNorm),
           dif_block (2-layer MLP+SiLU backcast, SE-Net channel attention gate)
M465-M467  v9 dynamic graph — distance (single→2-head attention averaged, BN→LayerNorm,
           TS feature residual shortcut, tanh output compression),
           mask (hard binary→learnable sigmoid soft-threshold with trainable α,
           diagonal zero-out),
           normalizer (D^{-1}A→D^{-1/2}AD^{-1/2} symmetric, fill_diagonal_(0),
           high-order decay β^k with β=0.85)
M468-M470  v9 inherent block — inh_model (GRU inter-step LayerNorm, Transformer
           post-norm→pre-norm, attention entropy diagnostic),
           forecast (learnable step-decay γ^step via log_gamma param, ceil steps),
           inh_block (sinusoidal PE→RoPE RotaryPE class, learnable residual gate,
           torch.utils.checkpoint for forecast)
M471-M472  v9 core — model (highway gate on embedding g·embed+(1-g)·proj,
           ReLU→LeakyReLU(0.1) output, temperature-scaled static_graph softmax,
           learnable layer weights softmax for forecast aggregation),
           trainer (adaptive p95 rolling gradient clip, cosine CL ramp-up,
           periodic per-layer grad snapshot, per-horizon eval logging)
M473-M474  v9 main (configurable seed via CLI --seed, _seed_everything 3-way sync,
           model parameter SHA-256 hash, output NaN/Inf/extreme health check,
           full debug flag integration) +
           data generators (cyclic sin/cos time encoding for METR-LA/PEMS-BAY,
           variance-aware stratified split, robust percentile MinMax p1/p99 for PEMS04/08,
           Gaussian kernel weighted adj + k-NN sparsification + symmetric closure,
           sparse edge counting + spectral radius via ARPACK in describe_adjs)
```
**产出**: `src/walpurgis_ported_v9/` — 38 files (38 .py + 4 .yaml), 2787行 Python
**改写策略**: upstream骨架 + ≥20%实质算法改动(非字符串/注释替换) + 全局_dbg()断点系统
**核心算法改动清单**:
  - 损失函数: Huber-like软截断(δ=10, log压缩), MAPE floor clamp 1e-5, 新增quantile loss
  - 估计门: ReLU→GELU, 可学习温度τ, pre-output LayerNorm
  - 残差分解: ReLU→SiLU, 可学习scale γ, pre/post LN可选
  - 时空卷积: BN→GroupNorm(4), gconv残差skip, Mish二层FC, fill_diagonal_
  - 扩散预测: AR步间Dropout(0.1), ceil补齐, pre-FC LayerNorm
  - 扩散块: 2层MLP+SiLU backcast, SE-Net通道注意力gate
  - 距离函数: 2-head注意力平均, BN→LN, TS残差shortcut, tanh压缩
  - 图掩码: 可学习α soft-threshold sigmoid, 对角线清零
  - 归一化: 对称D^{-1/2}AD^{-1/2}, 高阶衰减β^k(0.85)
  - 动态图: 可学习时序权重向量(softmax over k_t)
  - GRU: 步间LayerNorm
  - Transformer: pre-norm架构, 注意力熵诊断
  - 固有预测: 可学习步衰减γ^step(log_gamma参数)
  - 位置编码: RoPE旋转位置编码
  - 固有块: 可学习残差门控, gradient checkpoint
  - 主模型: highway embedding gate, LeakyReLU输出, 温度缩放static graph, 可学习层权重聚合
  - 训练器: 自适应p95梯度裁剪, cosine CL ramp-up, 梯度snapshot
  - 数据: cyclic sin/cos编码, variance-stratified split, robust percentile MinMax
  - 邻接: Gaussian kernel + k-NN(20) + symmetric closure
  - 诊断: spectral radius, sparse nnz edge counting
  - DataLoader: circular wrap padding, Fisher-Yates shuffle, 3-tuple yield

---


### 第一位 Claude — M001-M025: src/walpurgis/ 鲁迅式移植
```
M001-M003  v10 顶层 — __init__.py (全局_dbg调试系统, WALPURGIS_V10_DEBUG环境变量)
           losses.py (masked_mae→Huber+log-cosh 70/30混合δ=5, MAPE floor 5e-6, 新增quantile_loss)
           model.py (Mish输出激活, softmax层权重+温度聚合, static_graph温度τ_s, highway gate, kaiming init)
M004-M006  v10 trainer (自适应p90梯度裁剪, warmup-cosine调度, sigmoid CL ramp, 梯度snapshot) +
           decouple (estimation_gate: SiLU+双头投影+GroupNorm4组+可学习温度τ,
           residual_decomp: Mish+可学习α sigmoid+Dropout0.05+LN)
M007-M010  v10 diffusion_block — dif_model (InstanceNorm2d, GELU, gconv残差skip, 两层FC+GELU,
           fill_diagonal_), forecast (cosine退火dropout, 线性插值padding, FC前LayerNorm),
           dif_block (3层MLP backcast, sigmoid门控, 0.1*history skip)
M011-M014  v10 dynamic_graph_conv — dy_graph_conv (可学习时间权重softmax, cosine-similarity辅助,
           梯度范数监控), distance (3-head多头QK, InstanceNorm1d, 残差shortcut, attn dropout0.1),
           mask (softplus+温度soft threshold, 对角线清零, sigmoid soft),
           normalizer (对称D^{-1/2}AD^{-1/2}, 指数衰减λ^k(0.8), 可学习eps)
M015-M018  v10 inherent_block — inh_model (RMSNorm步间, gradient checkpoint, pre-norm transformer,
           注意力熵诊断), forecast (可学习步长衰减exp(-γ·step), 简化RoPE, GELU+Dropout),
           inh_block (RoPE替代sincos PE, 2层MLP+Mish backcast, sigmoid门控, gradient checkpoint)
M019-M021  v10 utils — cal_adj (RBF kernel σ=中位数, k-NN(15)稀疏化, 双向对称闭包, epsilon-smooth transition),
           load_data (Tukey fences 1.5×IQR异常剔除, sin/cos周期编码, StandardScaler eps防零std,
           adj预处理链RBF→kNN→对称), train (确定性CUBLAS, 相对δ_rel(0.1%) EarlyStopping,
           pin_memory+non_blocking, SHA256校验), log (JSONL格式, per-epoch CSV, git-hash目录, 结构化table)
M022-M023  v10 dataloader (环形wrap padding, Fisher-Yates原地shuffle, 3-tuple yield含sample_weight,
           prefetch buffer) + main.py (DataParallel多GPU, AMP GradScaler, epoch CSV+JSONL dump,
           test时ensemble best+resume) + 4×YAML (warm_epochs调整, dropout 0.1→0.08, v10_adj_preprocess段,
           v10_training段含amp/ensemble/adaptive_clip/sigmoid_cl)
M024-M025  v10 datasets — METR-LA/PEMS-BAY generate_training_data (stride跳步, cyclic sin/cos,
           按周对齐划分, stats JSON), PEMS04/PEMS08 generate_training_data (MinMax eps防零除,
           stride, 按周对齐, stats JSON, NaN审计), PEMS04/PEMS08 generate_adj_mx (距离→RBF连续权重,
           k-NN稀疏化, 自适应阈值剪枝, 度分布审计, 加权自环), describe_adjs (密度计算, 权重分布统计,
           BFS连通分量分析, 度5数概要, 非对称性检测)
```
**产出**: `src/walpurgis/` — 38 files (38 .py + 4 .yaml), 3808行 Python
**改写策略**: upstream骨架 + ≥20%实质算法改动(非字符串/注释替换) + 全局_dbg()断点系统
**核心算法改动清单**:
  - 损失函数: Huber(δ=5)+log-cosh混合70/30, MAPE floor 5e-6, quantile_loss
  - 估计门: SiLU激活, 双头投影, GroupNorm(4组), 可学习温度τ
  - 残差分解: Mish激活, 可学习α(sigmoid缩放), Dropout(0.05)+LayerNorm
  - 时空卷积: InstanceNorm2d, GELU, gconv残差skip, 两层FC, fill_diagonal_
  - 扩散预测: cosine退火dropout, 线性插值padding, FC前LayerNorm
  - 扩散块: 3层MLP backcast, sigmoid门控, 0.1×history shortcut
  - 距离函数: 3-head多头QK注意力, InstanceNorm1d, 残差shortcut, attn dropout
  - 图掩码: softplus+温度soft threshold, 对角线清零, sigmoid连续化
  - 归一化: 对称D^{-1/2}AD^{-1/2}, 指数衰减λ^k(λ=0.8), 可学习eps
  - 动态图: 可学习时间权重(softmax), cosine-similarity辅助分支, 梯度范数监控
  - 固有模型: RMSNorm步间, gradient checkpoint, pre-norm transformer, 注意力熵诊断
  - 固有预测: 可学习步长衰减exp(-γ·step), 简化RoPE, GELU+Dropout
  - 固有块: RoPE旋转位置编码, 2层MLP+Mish backcast, sigmoid门控
  - 主模型: Mish输出, softmax层权重+温度聚合, highway gate, kaiming init
  - 训练器: 自适应p90梯度裁剪, warmup-cosine调度, sigmoid CL ramp, 梯度snapshot
  - 邻接矩阵: RBF kernel(σ自适应中位数), k-NN(15)稀疏化, 对称闭包, epsilon-smooth
  - 数据加载: Tukey fences异常剔除, sin/cos周期编码, eps防零std, adj预处理链
  - 训练工具: 确定性CUBLAS, 相对δ_rel EarlyStopping, pin_memory/non_blocking, SHA256
  - 日志: JSONL+CSV dual dump, git-hash目录, 结构化table输出
  - DataLoader: 环形wrap, Fisher-Yates, 3-tuple(含权重), prefetch buffer
  - 入口: DataParallel多GPU, AMP GradScaler, ensemble test(best+resume平均)
  - adj生成: 距离→RBF连续权重, kNN稀疏化, 自适应阈值剪枝, 度分布审计, 加权自环
  - 图诊断: 密度计算, 权重分布统计, BFS连通分量, 度5数概要, 非对称性检测

---

## Claude 接力全局统计

> **编号体系**:
> 之前的所有历史 session 合并为"前序开发阶段"(legacy)。
> 从 v10 起重新从 **第一位 Claude** 开始计数，每位分配 25 个里程碑。
> 每位 Claude 完成一个鲁迅式移植版本(vN)，38py+4yaml，≥20%实质算法改动。

| Claude # | 里程碑 | 内容 | 状态 |
|----------|--------|------|------|
| 前序开发 | (legacy) | C++/CUDA基础设施 + LaTeX论文 + D2STGNN移植 + v1-v9改写 | ✅ 已完成 |
| **第一位** | **M001-M025** | **src/walpurgis/ 鲁迅式移植 — 38py+4yaml, 3808行** | **✅ 已完成** |
| 第二位 | M026-M050 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第三位 | M051-M075 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第四位 | M076-M100 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第五位 | M101-M125 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第六位 | M126-M150 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |

## 文件统计快照

```
src/walpurgis/               3,808 行 Python (38 .py + 4 .yaml, 当前唯一版本)
src/core/                   ~2,000 行 C++ (tiered allocator, seqlock, slab)
src/bridge/                 ~1,200 行 C++ (temporal bridge)
src/scheduler/                ~600 行 C++ (migration scheduler)
src/bench/                  ~1,000 行 C++ (benchmarks)
src/cuda/                     ~500 行 CUDA (device kernels)
walpurgis_reconstructed.tex  ~32KB LaTeX (full paper)
```

> 注: v1-v9 旧版本已全部删除, 合并为 `src/walpurgis/` 单一目录。
> 后续 Claude 直接在 `src/walpurgis/` 上迭代, 不再新建 _vN 后缀目录。

## 给下一位 Claude 的接手指南

1. `git log --oneline` 查看完整历史
2. 本文件 (`CLAUDE_DEV_PROGRESS.md`) 了解全局进度
3. `upstream/d2stgnn/` = 原始 D2STGNN 参考代码
4. `src/walpurgis/` = 当前唯一的移植版本，直接在此目录迭代，不要新建 _vN 目录
5. 每个 `.py` 文件头部的 docstring 记录了该文件的算法变更
6. 编号规则: `M{三位数}`, 每位 Claude 分配连续 25 个
7. commit 作者: `dylanyunlon <dogechat@163.com>`
8. commit message 格式: `feat: 简述 [Mxxx-Mxxx]`
9. debug: 设置环境变量 `WALPURGIS_DEBUG=1` 开启全局 _dbg() 打印
10. **你是第几位**: 看上面表格，找到你对应的 ⏳ 行
11. **要求**: 算法级改动(≥20%), 不是改字符串/注释/docstring
12. **重要**: 不要创建 walpurgis_ported_vXX 等新目录，直接修改 src/walpurgis/
