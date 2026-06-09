# Walpurgis Multi-Claude 开发计划

## 状态总结 (Claude-7 当前 — 本轮主控)

### SOTA 对比 (METR-LA, avg 12 horizons)
| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| TITAN | 2024 | 2.88 | 5.33 | — |
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| PDFormer | 2023 | 2.94 | 6.08 | 8.56% |
| STEP | 2022 | 2.98 | — | — |
| D2STGNN (upstream) | 2022 | 3.04 | 6.23 | 8.33% |
| **Walpurgis (当前最佳)** | — | **2.90** | **5.91** | **7.91%** |
| 目标 | — | **<2.85** | — | — |

### 服务器 (ags1)
- 2x A6000 (48GB) + 1x H100 NVL (96GB), AMD EPYC 9354 128核, ~1.5TB RAM
- CUDA 11.5, Driver 550.144, Conda: walking3
- GPU全在NUMA node1

### 铁律
- **不开新分支**, 不用 v2/v3/port 等后缀, 不用奇技淫巧后缀
- **改的是算法**: 不改字符串/docstring/str_replace表面功夫
- 所有改动直接 push 到 main
- 实验结果自动写入 experiments/results/ 并 push
- 服务器**只负责运行** experiments/run_server_experiment.sh

---

## Claude 任务分配

### 第一位 Claude (M001-M025) ✅ 已完成
**角色**: 项目审计 + 基础设施
- 分析项目结构和所有文件
- 诊断训练发散原因 (CL + config override + OOM)
- 更新 SOTA 对比表
- 创建自动化实验脚本

### 第二位 Claude (M026-M050) ✅ 已完成
**角色**: 修复训练发散 + 模型扩容
- 修复config覆盖bug
- 修复CL发散 (sigmoid ramp)
- 模型扩容适配H100

### 第三位 Claude (M051-M075) ✅ 已完成
**角色**: 算法改进 Phase 1
- SE Channel Attention + LN
- Exponential horizon loss
- LR schedule tuning
- Cascade residual mean-pool
- Depth gate init

### 第四位 Claude (M076-M100) ✅ 已完成
**角色**: 冲击 SOTA
- CosineAnnealingWarmRestarts
- Adaptive adjacency learning
- Data augmentation (node dropout + time shift)

### 第五位 Claude (M101-M125) ✅ 已完成
**角色**: 算法精炼 + OOM修复 + 断点调试 + 服务器实验流程
- 分块图卷积 (chunked gconv)
- 频率域残差注入
- 自适应温度LogCosh
- 增强断点调试框架
- 服务器实验脚本

### 第六位 Claude (M126-M150) ✅ 已完成
**角色**: LR调参
- CosineAnnealing T_0=30, eta_min=5e-6
- patience=50

### 第七位 Claude (M151-M175) ✅ 本轮主控 (当前)
**角色**: 算法进一步优化 + 子Claude调度
- Feature Refinement Module (gated refinement after cascade)
- Learnable horizon loss weights (70% fixed + 30% data-driven)
- Per-horizon MAE实时诊断
- v10后缀清理
- 派发Claude-8任务

### 第八位 Claude (M176-M200) ✅ 已完成 (子模型)
**角色**: 模型扩容 + 正则化
- num_hidden 96→112 (更多表达能力)
- 输出头Dropout(0.1) (防过拟合)
- horizon_scale 0.12→0.15 (更强调远horizon)
- SYNTH数据集快速验证通过

### 第九位 Claude (M201-M225) — 待执行
**角色**: 在服务器跑 METR-LA 完整实验
任务:
1. git pull origin main 获取所有算法改动
2. GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
3. 如果MAE<2.85: 胜利! 数据写入tex
4. 如果MAE在2.85-2.90: 微调horizon_scale或batch_size
5. 多种子评估: SEED=123 和 SEED=456

### 第十位 Claude (M226-M250) — 待执行
**角色**: TeX论文数据填充
1. 从 experiments/results/summary.json 提取最佳数据
2. 填入 walpurgis_reconstructed.tex 的实验表格
3. Per-horizon MAE/RMSE/MAPE 12行表格
4. SOTA对比表
5. 确保tex可编译

### 第十一位 Claude (M251-M275) — 待执行
**角色**: 消融实验
1. 逐个关闭: SE / cascade residual / depth gate / freq injection / FRM
2. 每个配置跑完push结果
3. 填入tex消融表

### 第十二位 Claude (M276-M300) — 待执行
**角色**: PEMS-BAY数据集实验 (如果METR-LA已SOTA)

---

## 服务器运行指南

```bash
# 在 ags1 上 (walking3 conda 环境)
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git pull origin main

# 设置token以自动push
export GIT_TOKEN=$GIT_TOKEN

# 标准实验 (H100, 200 epoch)
GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh

# 多种子
for SEED in 42 123 456; do
  SEED=$SEED GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
done

# 查看结果
cat experiments/results/summary.json
```

## 关键文件索引

| 文件 | 用途 |
|------|------|
| src/walpurgis/models/model.py | 主模型 (SE+cascade+depth gate+freq injection+FRM) |
| src/walpurgis/models/losses.py | 损失函数 (cascade_aware + adaptive LogCosh + learnable hw) |
| src/walpurgis/models/trainer.py | 训练循环 (CL sigmoid + diagnostics + learnable horizon) |
| src/walpurgis/models/diffusion_block/dif_model.py | 扩散图卷积 (chunked gconv) |
| src/walpurgis/configs/METR-LA.yaml | 超参数 (num_hidden=112) |
| train_walpurgis.py | 训练入口 |
| experiments/run_server_experiment.sh | 服务器全流程实验 |
| bench/sota.json | SOTA 对比 |
| walpurgis_reconstructed.tex | 论文 |

---

## Phase 2: SOTA冲刺 (M301-M450)

### 第一位 Claude (M301-M325) ✅ 已完成
**角色**: 算法移植 — 从TITAN/STAEformer鲁迅式拿法
- 从STAEformer移植: 自适应时空嵌入 (learnable [L,N,d] embedding)
- 从TITAN移植: PSD不确定性感知损失加权
- 从STAEformer移植: 输出头时序自注意力 (TemporalCrossAttention)
- 增强运行时诊断: 每200步打印完整模型状态
- SYNTH数据集端到端验证通过
- 创建Phase 2全部子Claude任务文件

### 第二位 Claude (M326-M350) — 待执行
**角色**: 服务器METR-LA完整实验
- SYNTH快速验证 → METR-LA 200 epoch完整训练
- 观察adaptive_emb_gate / uncertainty_mean诊断
- 多种子评估 (42, 123, 456)
- 任务文件: tasks/task_claude2_M326_M350.md

### 第三位 Claude (M351-M375) — 待执行
**角色**: TeX数据填充
- 提取实验结果 → 填入论文表格
- 更新SOTA对比表 + per-horizon表
- 确保pdflatex编译通过
- 任务文件: tasks/task_claude3_M351_M375.md

### 第四位 Claude (M376-M400) — 待执行
**角色**: 消融实验
- 逐个关闭: adaptive_emb / uncertainty / temporal_attn / SE / cascade
- 记录每个配置的MAE变化
- 填入论文消融表
- 任务文件: tasks/task_claude4_M376_M400.md

### 第五位 Claude (M401-M425) ✅ 已完成
**角色**: 多种子评估 (SYNTH)
- SEED=123: best_val_MAE=6.6784 (Δ+1.6422 vs seed42, +32.6%)
- SEED=456: best_val_MAE=5.7798 (Δ+0.7436 vs seed42, +14.8%)
- mean±std across seeds 123+456: 6.2291 ± 0.4493 (3 epoch CPU, SYNTH)
- 结果写入: experiments/results/multi_seed.json
- 备注: SYNTH 3-epoch CPU run显示种子间有波动属正常; 服务器200 epoch完整训练方可评估真实稳定性
- 任务文件: tasks/task_claude5_M401_M425.md

### 第六位 Claude (M426-M450) — 待执行
**角色**: 论文最终收尾
- 汇总所有数据 → 完善tex
- pdflatex编译验证
- 最终commit + push
- 任务文件: tasks/task_claude6_M426_M450.md

---

## Phase 2 算法改动总结

### 改动1: 自适应时空嵌入 (from STAEformer)
**文件**: src/walpurgis/models/model.py
**原理**: 可学习的 [L, N, d_adp] 参数矩阵，每个时间步每个节点有独立的嵌入向量。
通过门控 (sigmoid gate) 注入到embedding后的特征中。
**预期效果**: 捕获节点特异性的时序模式，STAEformer论文显示该组件贡献~0.1 MAE。

### 改动2: PSD不确定性感知损失加权 (from TITAN)
**文件**: src/walpurgis/models/losses.py
**原理**: 对label序列做时间差分 → 计算变异系数(CV) → 高CV=可预测，低CV=不确定。
不确定样本降权40%，防止模型被噪声样本带偏。epoch<20不启用。
**预期效果**: 更稳定的训练，减少远horizon过拟合噪声。

### 改动3: 输出头时序自注意力 (from STAEformer)
**文件**: src/walpurgis/models/model.py (TemporalCrossAttention class)
**原理**: 在cascade聚合后、regression前，对时间维度做2-head self-attention。
捕获输出序列步间的依赖关系（e.g. h3和h6的关联）。
**预期效果**: 更一致的多步预测，减少远horizon退化。
