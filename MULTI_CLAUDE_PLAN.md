# Walpurgis Multi-Claude Development Plan

> 目标: METR-LA 上超越 TITAN (MAE 2.52@h3) 和 STAEFormer (MAE 2.90 avg)
>
> 服务器: ags1 — 2×A6000 (48GB) + 1×H100 NVL (96GB)
>
> 仓库: github.com/dylanyunlon/walpurgis-WTFGG (main 分支, 不开新分支)

## 里程碑分配

### 第一位 Claude (M100-M109) ✅ 完成
**任务: 项目审计 + 实验基础设施**

| ID | 内容 | 状态 |
|----|------|------|
| M100 | 项目架构审查, SOTA 差距分析 | ✅ |
| M101 | 清理仓库: 删除旧变体后缀/stale checkpoints/logs | ✅ |
| M102 | 修复 METR-LA config: epochs 3→100, patience 15 | ✅ |
| M103 | 创建 experiments/ 实验流水线 (setup/run/eval/push) | ✅ |
| M104 | 更新 .gitignore, README, 本计划文档 | ✅ |
| M105 | 创建基线对比脚本 (D2STGNN upstream + STAEFormer) | ✅ |
| M106 | Git push 所有清理和基础设施 | ✅ |
| M107-M109 | 预留 | - |

### 第二位 Claude (M110-M119)
**任务: 在服务器上实际运行实验, 获取基线数据**

关键指令:
1. `git pull origin main` 拉取最新代码
2. `bash experiments/setup_env.sh` 配置环境
3. `bash experiments/run_baselines.sh --model d2stgnn --gpu 0` 跑 D2STGNN 基线
4. 验证 METR-LA 数据正确加载 (207 nodes, 34272 timesteps)
5. 确认 upstream D2STGNN 基线在 METR-LA 上复现 MAE ≈ 3.04

| ID | 内容 |
|----|------|
| M110 | 服务器环境配置, conda activate, 数据下载 |
| M111 | 运行 upstream D2STGNN 基线, 100 epochs, 验证 MAE≈3.04 |
| M112 | 运行 Walpurgis (Cascade变体) 100 epochs on METR-LA |
| M113 | 对比 Cascade vs D2STGNN baseline, 诊断差距 |
| M114 | 结果 JSON push 到 experiments/results/ |
| M115-M119 | 预留 |

### 第三位 Claude (M120-M129)
**任务: 算法改进 — 修复 mode collapse, 提升到 baseline 水平**

根据第二位 Claude 的诊断数据, 修改核心算法:
- `src/walpurgis/models/model.py` — 模型架构调整
- `src/walpurgis/models/losses.py` — 损失函数修复
- `src/walpurgis/models/trainer.py` — 训练策略优化

重点修改方向 (根据 variant_analysis.json 排名):
1. **Cascade SE+Dense skip** — 已实现, 需调参
2. **损失函数**: Huber+LogCosh→纯 MAE (upstream 使用 masked_mae)
3. **学习率**: 对齐 upstream 的 MultiStepLR schedule
4. **Curriculum learning**: 正确实现 (upstream 有, 当前可能 broken)

| ID | 内容 |
|----|------|
| M120 | 分析 M112 的训练日志, 定位 mode collapse 根因 |
| M121 | 修复损失函数: 从 Cascade loss 回退到 masked_mae |
| M122 | 对齐 upstream 训练策略 (lr_schedule, curriculum) |
| M123 | 运行修复后的模型 100ep, 目标 MAE < 3.10 |
| M124 | Push 修复代码 + 结果 |
| M125-M129 | 预留 |

### 第四位 Claude (M130-M139)
**任务: 算法创新 — 超越 D2STGNN baseline**

在 M120-M124 的稳定基础上, 逐一引入创新:
1. SE channel attention (验证有效再保留)
2. Dense cascade connections (验证有效再保留)
3. Dynamic depth gating (验证有效再保留)
4. 每加一个改动跑一次实验, 只保留有效的

目标: **MAE < 2.96** (超越 STG-NCDE)

| ID | 内容 |
|----|------|
| M130 | +SE attention 消融实验 |
| M131 | +Dense cascade 消融实验 |
| M132 | +Dynamic depth 消融实验 |
| M133 | 组合最优改动, 100ep 训练 |
| M134 | Push 结果 + 更新 comparison_table |
| M135-M139 | 预留 |

### 第五位 Claude (M140-M149)
**任务: 冲击 SOTA — 超越 STAEFormer**

目标: **MAE < 2.90** (METR-LA avg)

在 M130 的最优组合上, 尝试高潜力改动:
- Multi-view fusion (Prism 变体思路)
- Fourier positional encoding
- 更深的 submodule 重写
- 多 seed 验证稳定性

| ID | 内容 |
|----|------|
| M140 | 引入 Fourier/sin-cos 时间编码 |
| M141 | 引入 multi-scale feature fusion |
| M142 | 多 seed (42/137/271) 验证 |
| M143 | 100ep 全量训练, 最终数据 |
| M144 | Push SOTA 结果 |
| M145-M149 | 预留 |

### 第六位 Claude (M150-M159)
**任务: 论文 TeX 填充 + PEMS-BAY 实验**

1. 将 experiments/results/ 中的数据填入 tex
2. 在 PEMS-BAY 上复现 (证明泛化性)
3. 完成 comparison table, 消融实验表
4. 生成 training curve 图

| ID | 内容 |
|----|------|
| M150 | PEMS-BAY 实验 (100ep) |
| M151 | 更新 walpurgis_reconstructed.tex 中的实验数据 |
| M152 | 生成 comparison_table.tex (与 TITAN/STAEFormer/D2STGNN) |
| M153 | 生成消融实验表 (ablation study) |
| M154 | 生成 training curve 图 |
| M155-M159 | 预留 |

## 服务器操作规范

### Git 工作流
```bash
# 每次开始前
git pull origin main

# 工作完成后
git add -A
git commit -m "M1XX: 简要描述"
git push origin main
```

### GPU 分配
| GPU | 型号 | 显存 | 用途 |
|-----|------|------|------|
| 0 | A6000 | 48GB | 基线实验 / 消融实验 |
| 1 | A6000 | 48GB | 基线实验 / 消融实验 |
| 2 | H100 NVL | 96GB | 主实验 (Walpurgis) |

### Conda 环境
```bash
conda activate walpurgis
# 或 conda activate base (如果 base 已有 pytorch)
```

### 关键路径
```
src/walpurgis/models/model.py    — 模型架构 (改这里)
src/walpurgis/models/losses.py   — 损失函数 (改这里)
src/walpurgis/models/trainer.py  — 训练循环 (改这里)
src/walpurgis/configs/METR-LA.yaml — 超参数配置
train_walpurgis.py               — 训练入口
experiments/results/              — 实验结果 (自动 push)
```

### 不要做的事
- ❌ 开新分支 (只用 main)
- ❌ 创建带后缀的目录 (walpurgis_xxx, v2, port 等)
- ❌ 改 docstring / str_replace 字符串
- ❌ 在 SYNTH 上做实验 (只用 METR-LA / PEMS-BAY)
- ❌ 训练少于 50 epochs

## SOTA 参考数据 (METR-LA, Average across horizons)

| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| TITAN | 2024 | 2.88 | — | — |
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| PDFormer | 2023 | 2.94 | 6.08 | 8.56% |
| STG-NCDE | 2022 | 2.96 | 6.51 | 9.13% |
| D2STGNN | 2022 | 3.04 | 6.23 | 8.33% |
| **Walpurgis (目标)** | **2026** | **< 2.85** | **< 5.80** | **< 8.00%** |
