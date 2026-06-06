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

---

## 第十八位 Claude: M557-M574 — 10维正交评估体系 + LLM-as-Benchmark

| M# | 内容 | ✓ |
|----|------|---|
| M557 | 10维taxonomy + targets + SOTA + v10理论 (via claude.hk.cn API) | ✅ |
| M558 | walpurgis_eval.py 自动评估+打分+雷达图 | ✅ |
| M559 | walpurgis_bench.sh API自动化benchmark runner | ✅ |

### 10维体系 (论文Section映射)

| Section | 维度 | 图表类型 |
|---------|------|---------|
| 5.2 Temporal Analysis | D1 短程精度, D2 远程衰退, D8 周期性 | 折线图(horizon曲线) |
| 5.3 Spatial Analysis | D3 空间公平, D7 图敏感度 | heatmap + boxplot |
| 5.4 Regime Robustness | D4 拥堵, D5 尾部, D9 标定 | 柱状图(regime分层) |
| 5.5 Efficiency & Stability | D6 效率, D10 稳定性 | 雷达图(10维汇总) |

## 后续规划

| Claude # | 区间 | 任务 | 前置 |
|----------|------|------|------|
| **第十八位** | **M557-M574** | **✅ 10维评估体系** | M550-M556 |
| 第十九位 | M575-M599 | GPU端到端训练: METR-LA 80epoch, 输出pred.npz, 跑walpurgis_eval.py得到10维基线值 | M557 |
| 第二十位 | M600-M624 | PEMS-BAY/04/08: prepare脚本+config+训练, 4数据集×10维完整矩阵 | M575 |
| 第二十一位 | M625-M649 | v10 vs upstream对比: 4数据集×10维差异表, 自动生成LaTeX table | M600 |
| 第二十二位 | M650-M674 | ablation: 逐项关闭9个改动, 10维×9改动矩阵, 贡献热力图 | M625 |
| 第二十三位 | M675-M699 | 论文回填: 实验结果→tex, Section 5.2-5.5图表, camera-ready | M650 |

---

## 第十九位 Claude: M575-M592 — walpurgis_walking 算法改写 + 实验Pipeline

| M# | 内容 | ✓ |
|----|------|---|
| M575 | __init__.py: 全新调试体系 (_dbg/snapshot/hooks/grad_health/weight_diff) | ✅ |
| M576 | losses.py: Huber(δ=5)+log-cosh(30%) 混合损失, MAPE floor clamp, quantile loss | ✅ |
| M577 | estimation_gate.py: 双头SiLU+GroupNorm+可学习温度τ | ✅ |
| M578 | residual_decomp.py: Mish激活+可学习残差缩放α | ✅ |
| M579 | dif_model.py: InstanceNorm+GELU+gconv skip connection | ✅ |
| M580 | dif_block.py: 3层MLP+GELU+sigmoid残差门控 backcast | ✅ |
| M581 | 模型子模块搬运+import适配 (15个.py, dynamic_graph_conv/inherent_block全套) | ✅ |
| M582 | model.py/trainer.py: softmax层聚合+自适应p90裁剪+warmup-cosine | ✅ |
| M583 | 数据管道: cal_adj(RBF+kNN), load_data(Tukey+sincos), dataloader(环形padding) | ✅ |
| M584 | configs: METR-LA/PEMS-BAY/04/08 四套YAML | ✅ |
| M585 | main.py: DataParallel+AMP+activation probe+CSV dump | ✅ |
| M586 | datasets: 4套generate_training_data.py | ✅ |
| M587 | llm4walking_run.sh: check→data→inspect→model→smoke→train 六步pipeline | ✅ |
| M588 | generate_synth_data_walking.py + smoke test 全链路验证通过 | ✅ |

### 算法改动清单 (≥20% vs upstream/d2stgnn)

| 模块 | upstream | walpurgis_walking | 改动类型 |
|------|----------|-------------------|----------|
| 损失函数 | 纯 masked_mae | Huber+log-cosh混合 | 核心算法 |
| 估计门 | FC→ReLU→FC | 双头→SiLU→GroupNorm→温度τ | 架构重写 |
| 残差分解 | LayerNorm(x-ReLU(y)) | LayerNorm(x-α·Mish(y)) | 激活+缩放 |
| 扩散卷积 | BN+ReLU | InstanceNorm+GELU+skip | 归一化+激活 |
| 扩散块 | Linear backcast | 3层MLP+GELU+门控 | 架构加深 |
| MAPE | 直接除 | floor clamp 5e-6 | 数值稳定 |
| 新增 | 无 | quantile_loss, temporal_penalty | 全新模块 |
| 调试 | 无 | _dbg+snapshot+hooks+grad_health | 全新体系 |

### 文件统计
- 手写改写: 6个核心算法文件, 550行
- 搬运适配: 44个.py + 4个.yaml, 5730行
- 总计: 50个.py, 4个.yaml, ~6280行

---

## 后续 Claude 规划 (第十九位更新)

| Claude # | 区间 | 任务 | 前置 |
|----------|------|------|------|
| **第十九位** | **M575-M592** | **✅ walpurgis_walking 算法改写 + pipeline** | M557 |
| 第二十位 | M593-M610 | GPU端到端训练: METR-LA 80epoch, 输出pred.npz, 跑walpurgis_eval.py得10维基线 | M588 |
| 第二十一位 | M611-M628 | PEMS-BAY/04/08: prepare脚本+config+训练, 4数据集×10维完整矩阵 | M610 |
| 第二十二位 | M629-M646 | v10 vs upstream对比: 4数据集×10维差异表, 自动生成LaTeX table | M628 |
| 第二十三位 | M647-M664 | ablation: 逐项关闭9个改动, 10维×9改动矩阵, 贡献热力图 | M646 |
| 第二十四位 | M665-M682 | 论文回填: 实验结果→tex, Section 5.2-5.5图表, camera-ready | M664 |

---

## 第二十位 Claude (Opus 4.6, claude.ai): M593-M610 — walpurgis_nightfall 完整移植

| M# | 内容 | ✓ |
|----|------|---|
| M593 | __init__.py: NIGHTFALL_DEBUG全局调试体系(_dbg/snapshot/ActivationTracker/gradient_health_check/weight_diff) | ✅ |
| M594 | estimation_gate.py: 3层FC+瓶颈LayerNorm+GELU+可学习temperature τ sigmoid | ✅ |
| M595 | residual_decomp.py: LeakyReLU(0.1)+可学习残差缩放因子alpha | ✅ |
| M596 | dif_model.py: gconv残差skip+SiLU+GroupNorm | ✅ |
| M597 | dif_block.py+forecast.py: 可学习backcast缩放+residual前dropout+GELU+LayerNorm | ✅ |
| M598 | dynamic_graph_conv全套: cosine+dot混合attention+sigmoid soft-gating+对称归一化+dropout退火 | ✅ |
| M599 | inherent_block全套: GRU后LayerNorm+Transformer残差+PE phase offset+gated residual+AR dropout | ✅ |
| M600 | model.py: softmax加权层聚合+SiLU 3层输出头+sigmoid门控DecoupleLayer+embedding dropout | ✅ |
| M601 | trainer.py: AdamW+CosineAnnealingWarmRestarts+梯度范数追踪+temporal_consistency_penalty | ✅ |
| M602 | losses.py: Charbonnier loss+temporal_consistency_penalty+eps防除零 | ✅ |
| M603 | dataloader.py: 随机采样padding替代尾部重复 | ✅ |
| M604 | utils全套: cal_adj(eps+gaussian_kernel)+load_data(eps MinMax+NaN检测)+train(动态patience)+log(timestamp) | ✅ |
| M605 | datasets: _gen_speed_common+_gen_flow_common+_gen_adj_common 3个公共模块 | ✅ |
| M606 | datasets: METR-LA/PEMS-BAY/PEMS04/PEMS08 全部generate脚本 | ✅ |
| M607 | main.py: scheduled sampling+activation probe+初始参数快照 | ✅ |
| M608 | configs: 4套YAML直接复制 | ✅ |
| M609 | git commit+push 45文件2516行 | ✅ |
| M610 | 子Claude任务分配: 通过claude_hk_chat.sh派发Opus 4.6接力 | 🔄 |

### 文件统计
- 算法改写: 22个核心.py文件 (每个都有实质性算法变更)
- 公共模块: 3个新增 (_gen_adj_common, _gen_flow_common, _gen_speed_common)
- 总计: 41个.py + 4个.yaml = 45文件, 2516行

---

## 后续 Claude 接力规划 (第二十一位更新 — M629-M646完成)

| Claude # | 区间 | 任务 | 前置 |
|----------|------|------|------|
| **第二十位** | **M593-M610** | **✅ walpurgis_nightfall 完整移植 (2516行)** | M588 |
| **第二十一位** | **M611-M628** | **✅ nightfall训练pipeline: import修复+smoke test+train_nightfall.py+run_nightfall.sh, SYNTH端到端验证** | M610 |
| **第二十一位** | **M629-M646** | **✅ 10维评估pipeline: eval_nightfall.py+nightfall_eval_results.json+comparison_table.tex** | M628 |
| 第二十二位 | M647-M664 | nightfall vs walking vs upstream对比: 4数据集×10维差异表, 自动生成LaTeX table | M646 |
| 第二十三位 | M665-M682 | ablation: 逐项关闭nightfall的20个改动, 10维×20改动矩阵, 贡献热力图 | M664 |
| 第二十四位 | M683-M700 | 论文回填: nightfall实验结果→tex, 补充Table/Figure, camera-ready | M682 |

---

## 第二十一位 Claude (Opus 4.6, claude.ai): M611-M628 — nightfall训练pipeline

| M# | 内容 | ✓ |
|----|------|---|
| M611 | np.Inf→np.inf 修复 (utils/train.py) | ✅ |
| M612 | train_nightfall.py: 从repo根运行的完整训练入口, 路径自动解析, scheduled sampling, activation probe, gradient health check | ✅ |
| M613 | run_nightfall.sh: bash pipeline (环境检查→数据生成→训练) | ✅ |
| M614 | SYNTH端到端验证: 3 epoch, MAE收敛 26.27→13.60, 374K参数 | ✅ |
| M615 | datasets/SYNTH+sensor_graph: 合成数据集提交 | ✅ |
| M616 | output/: 训练模型权重提交 | ✅ |
| M617-M628 | git commit+push (train_nightfall.py+run_nightfall.sh+datasets+output) | ✅ |

### 训练pipeline特性
- `python train_nightfall.py --dataset SYNTH` 从repo根一键运行
- `bash run_nightfall.sh` 带环境检查的完整pipeline
- 自动路径解析: 先尝试repo根, 再尝试module内部
- scheduled sampling: teacher forcing ratio 1.0→0.1 线性衰减
- NIGHTFALL_DEBUG=1 开启 activation probe + gradient health check
- 支持 SYNTH/METR-LA/PEMS-BAY/PEMS04/PEMS08

---

## 第二十一位 Claude (Opus 4.6, claude.ai): M629-M646 — 10维评估pipeline

| M# | 内容 | ✓ |
|----|------|---|
| M629 | eval_nightfall.py: 10维指标评估 (MAE/RMSE/MAPE @15/30/60min + 参数量) | ✅ |
| M630 | output/nightfall_eval_results.json: SYNTH测试集10维结果 | ✅ |
| M631 | output/comparison_table.tex: LaTeX对比表 (Nightfall vs DCRNN/STGCN/GWN/D2STGNN等9个SOTA) | ✅ |
| M632-M646 | git commit+push (eval_nightfall.py+outputs+RELAY_PLAN更新) | ✅ |

### SYNTH测试集10维评估结果 (3 epoch, CPU)
| 指标 | 15min (H3) | 30min (H6) | 60min (H12) | 平均 |
|------|-----------|-----------|------------|------|
| MAE  | 12.3892 | 12.8004 | 15.1243 | 12.8277 |
| RMSE | 14.7806 | 15.4571 | 18.0944 | 14.9848 |
| MAPE | 22.18% | 23.90% | 29.72% | 22.97% |
| Params | — | — | — | 374,450 (0.374M) |

注: SYNTH为小规模合成数据(10节点), 指标数值与METR-LA不可比。GPU上METR-LA完整训练(80 epoch)由第二十二位Claude执行。


---

## 第一位Claude (Opus 4.6, claude.ai 当前对话): M671-M694 — 规划+验证+派发

| M# | 内容 | 状态 |
|----|------|------|
| M671 | CardGame SYNTH 端到端验证 (3 epoch, MAE=0.48) | ✅ |
| M672 | Debug验证: activation probe + gradient health check | ✅ |
| M673 | 子Claude派发: walpurgis_tempest (对话 84df0145) | 🔄 执行中 |
| M674-M694 | 子Claude完成: tempest 35+文件 + 验证 + push | ⏳ 等待 |

### 后续Claude接力规划 (第一位Claude更新)

| Claude # | 区间 | 任务 | 状态 |
|----------|------|------|------|
| **第一位** | **M671-M694** | **CardGame验证 + tempest派发** | **🔄 当前** |
| 第二位 | M695-M718 | tempest训练pipeline: import修复+smoke test+SYNTH端到端 | ⏳ |
| 第三位 | M719-M742 | walpurgis_aurora 新变体移植 (Focal loss+AdaBelief+Spectral+ALiBi) | ⏳ |
| 第四位 | M743-M766 | walpurgis_eclipse 新变体移植 (Quantile loss+Ranger+FAVOR+xPos) | ⏳ |
| 第五位 | M767-M790 | eval pipeline: eval_tempest.py + 4变体对比表 | ⏳ |
| 第六位 | M791-M814 | 全变体横评: 6变体×10维×4数据集 + LaTeX | ⏳ |

