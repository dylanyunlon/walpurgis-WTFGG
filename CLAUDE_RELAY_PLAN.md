# Walpurgis-WTFGG: Claude 接力开发进度 & 规划

> 截至第十位 Claude (当前)，总览全局。
> 每位 Claude 接手时读此文件 + `git log --oneline -20` 即可续接。

---

## 已完成

| Claude # | 里程碑区间 | 内容摘要 | 状态 |
|----------|-----------|---------|------|
| 第一位 | M001-M014 | C++/CUDA 底层基础设施 (tiered alloc, seqlock, slab, migration) | ✅ |
| 第二位 | M001-M075 | 论文写作 LaTeX (Introduction → Experimental Design) | ✅ |
| 第三位 | M101-M200 | D2STGNN 首次移植 + v1 改写 (`src/walpurgis/`) | ✅ |
| 第四位 | M201-M255 | v2 全量改写 (TensorProbe, Charbonnier, adaptive clip) | ✅ |
| 第五位 | M256-M274 | v3 全量改写 (6644行) | ✅ |
| 第六位 | M275-M299 | v4 全量改写 (MoE, SwiGLU, spectral SVD, 7632行) | ✅ |
| 第七位 | M300-M324 | 鲁迅式 v2 移植 upstream→`walpurgis_ported_v2` (2615行) | ✅ |
| 第八位 | M325-M349 | 鲁迅式 v3 移植 upstream→`walpurgis_ported_v3` (2202行) | ✅ |
| 第九位 | M350-M374 | 生成 `git am` patch, 作者归属, 本规划文档 | ✅ |
| **第十位** | **M375-M399** | **鲁迅式 v4 移植 upstream→`walpurgis_ported_v4` (3519行, 40文件, 25个有debug)** | **✅ 当前** |

---

## 规划: 下一批 Claude 的里程碑分配

| Claude # | 里程碑区间 | 任务方向 | 预估产出 |
|----------|-----------|---------|---------|
| 第十一位 | M400-M424 | 鲁迅式 v5 移植: 补全 `walpurgis_ported_v5` 剩余 8 个文件 (main.py, trainer.py, 6个datasets generators) + configs，在v4基础上再做20%变形 | ~1500行新增, v5完整化 |
| 第十二位 | M425-M449 | 统一测试框架: 为 v2/v3/v4/v5 写 `pytest` 单元测试, mock数据, 确保所有版本可 `import`, debug flag 冒烟测试, CI `Makefile` | tests/ 目录, ~2000行 |
| 第十三位 | M450-M474 | 实验管道: `run_experiment.py` 端到端脚本, 自动对比 v2/v3/v4/v5 在 METR-LA/PEMS-BAY 上的 MAE/RMSE/MAPE, 生成 LaTeX 对比表 | scripts/, ~1500行 |
| 第十四位 | M475-M499 | 论文实验章节补全: 把实验结果回填 `walpurgis_reconstructed.tex`, 补 Table 4-7 + Figure 5-8, ablation study 节 | .tex 更新 |
| 第十五位 | M500-M524 | C++ ↔ Python bridge: pybind11 暴露 `src/core/` tiered allocator 给 Python 训练循环, 跑通 heterogeneous memory 分配路径 | src/pybind/, ~2000行 |
| 第十六位 | M525-M549 | 性能调优 + profiling: 各版本训练速度对比, GPU utilization, memory peak tracking, bottleneck 定位报告 | perf/ 目录 + 分析报告 |

---

## v4 移植 (第十位 Claude) 算法改动清单

### 架构层
- EstimationGate: 2层→3层FC + LayerNorm瓶颈
- ResidualDecomp: ReLU→LeakyReLU(0.1)
- Distance attention: 可学习 temperature 参数
- Mask: 可学习 sigmoid soft-gating (nn.ParameterList)
- Normalizer: 行归一化 D⁻¹A → 对称归一化 D⁻¹/²AD⁻¹/²
- STLocalizedConv: 残差 skip (out += X_0)
- Diffusion forecast: GELU → projection
- RNNLayer: GRU后接LayerNorm
- TransformerLayer: 残差连接 (out = in + attn)
- Inherent forecast: scheduled sampling + AR dropout
- InhBlock: 可学习PE phase offset + gated residual backcast
- DecoupleLayer: sigmoid门控 alpha blending
- D2STGNN: softmax加权层聚合 + SiLU输出头

### 优化器层
- AdamW 替代 Adam
- CosineAnnealingWarmRestarts 替代 MultiStepLR
- 梯度范数追踪 (clip前)
- main.py: scheduled sampling 线性衰减

### 数据管道层
- Gaussian kernel 邻接矩阵加权 (PEMS04/08)
- eps-guarded MinMax 归一化
- NaN 检测 (windowing 前)
- 数据泄漏断言 (split 验证)

---

## 代码库文件统计 (第十位 Claude 完成后)

```
src/walpurgis/               — v1 原版 (第三位)       ~3,500 行
src/walpurgis_ported/        — v2 改写 (第四位)       ~4,500 行
src/walpurgis_ported_v2/     — 鲁迅式 v2 port (第七位) 2,615 行
src/walpurgis_ported_v3/     — 鲁迅式 v3 port (第八位) 2,202 行
src/walpurgis_ported_v4/     — 鲁迅式 v4 port (第十位) 3,309 行  ← NEW
src/walpurgis_ported_v5/     — 鲁迅式 v5 port (部分)   1,803 行 (27/39文件)
src/core/                    — C++ 底层               ~2,000 行
src/bridge/                  — C++ temporal bridge     ~1,200 行
src/scheduler/               — C++ migration          ~600 行
src/bench/                   — C++ benchmarks         ~1,000 行
src/cuda/                    — CUDA kernels           ~500 行
upstream/d2stgnn/            — 原始参考代码            2,822 行
```

## 给下一位 Claude 的操作手册

### 1. 快速接手
```bash
git log --oneline -20          # 看历史
cat CLAUDE_RELAY_PLAN.md       # 看本文件, 找到你的里程碑区间
cat CLAUDE_DEV_PROGRESS.md     # 看详细技术记录
tree src/ -L 2 --charset ascii # 看目录结构
```

### 2. 提交规范
```
feat(vN): 简要描述 [Mxxx-Mxxx]
```
作者: `dylanyunlon <dogechat@163.com>`

### 3. 生成 patch
```bash
git format-patch origin/main --stdout > your_patch.patch
# 或单个 commit:
git format-patch -1 HEAD --stdout > your_patch.patch
```

### 4. 应用 patch (用户侧)
```bash
git am < patch_file.patch
```

### 5. v4 Debug 开关
v4 使用全局 `_V4_DEBUG` flag, 运行前设环境变量关闭:
```python
# 在代码顶部: _V4_DEBUG = True
# 所有 debug 输出到 stderr, 不污染 stdout
# 25/35 个 .py 文件带有 debug 插桩
```

### 6. v5 剩余文件清单 (给第十一位)
```
datasets/raw_data/METR-LA/generate_training_data.py
datasets/raw_data/PEMS-BAY/generate_training_data.py
datasets/raw_data/PEMS04/generate_adj_mx.py
datasets/raw_data/PEMS04/generate_training_data.py
datasets/raw_data/PEMS08/generate_adj_mx.py
datasets/raw_data/PEMS08/generate_training_data.py
main.py
models/trainer.py
configs/ (4 yaml)
__init__.py (顶层)
```

---

## 第十七位 Claude: M550-M556 — LLM4Walking 实验运行 Pipeline

| M# | 内容 | ✓ |
|----|------|---|
| M550 | fix: InstanceNorm1d/cos_proj/np.Inf | ✅ |
| M551 | refactor: DataLoader索引化 | ✅ |
| M552 | fix: main.py生产化 | ✅ |
| M553 | config: METR-LA双环境 | ✅ |
| M554 | prepare_metrla.sh | ✅ |
| M555 | run_walpurgis.sh | ✅ |

## 后续

| Claude # | 区间 | 任务 |
|----------|------|------|
| 第十八位 | M575-M599 | GPU训练80epoch |
| 第十九位 | M600-M624 | PEMS-BAY/04/08 |
| 第二十位 | M625-M649 | 对比表 |
| 第二十一位 | M650-M674 | ablation |
| 第二十二位 | M675-M699 | 论文回填 |
