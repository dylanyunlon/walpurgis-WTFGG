# Walpurgis Multi-Claude 开发计划

## 状态总结 (Claude-5 当前)

### SOTA 对比 (METR-LA, avg 12 horizons)
| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| TITAN | 2024 | 2.88 | 5.33 | — |
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| PDFormer | 2023 | 2.94 | 6.08 | 8.56% |
| STEP | 2022 | 2.98 | — | — |
| D2STGNN (upstream) | 2022 | 3.04 | 6.23 | 8.33% |
| **Walpurgis (prev best)** | — | **3.08** | **6.04** | **8.17%** |
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
- 结果: OOM (batch_size过大)

### 第五位 Claude (M101-M125) ← 当前
**角色**: 算法精炼 + OOM修复 + 断点调试 + 服务器实验流程

已完成:
1. **分块图卷积** (dif_model.py): gconv中matmul分chunk执行, 降低内存峰值
2. **频率域残差注入** (model.py): DecoupleLayer的cascade_proj后加入FFT频率成分
3. **自适应温度LogCosh** (losses.py): 训练进度驱动温度退火 (早期平滑→后期精确)
4. **增强断点调试** (trainer.py): forward/loss/backward三阶段全状态dump
5. **服务器实验脚本** (experiments/run_server_experiment.sh): 自动训练→评估→push
6. **子Claude调度** (claude_hk_chat.sh): 通过claude.hk.cn派发任务给Sonnet 4.6

待在服务器执行:
```bash
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git pull origin main
GPU=2 EPOCHS=200 nohup bash experiments/run_server_experiment.sh &
```

### 第六位 Claude (M126-M150) — 待执行
**角色**: SOTA冲刺 — 根据M101-M125实验结果调参

任务:
1. 拉取 experiments/results/ 最新实验数据
2. 分析per-horizon MAE分布, 定位薄弱horizon
3. 针对性调整:
   - 如果远horizon(9-12)差: 加大horizon_scale
   - 如果近horizon(1-4)差: 降低CL warm_epochs
   - 如果整体偏高: 扩大模型(hidden 64→96, 仍然用分块gconv控制内存)
4. 多种子评估: seed=42,123,456
5. 跑完push结果

启动方式 (通过claude_hk_chat.sh):
```bash
bash claude_hk_chat.sh "你是第六位Claude(M126-M150)。
clone github.com/dylanyunlon/walpurgis-WTFGG, apt install tree, 
查看 experiments/results/ 下最新的 result.json,
分析METR-LA实验数据, 针对薄弱horizon做算法调整。
铁律: 不开分支、不用后缀、改算法不改字符串。
改完后 GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
作者: dylanyunlon <dogechat@163.com>
Token: <GIT_TOKEN_FROM_ENV>"
```

### 第七位 Claude (M151-M175) — 待执行
**角色**: TeX论文数据填充

1. 从 experiments/results/summary.json 提取最佳数据
2. 填入 walpurgis_reconstructed.tex 的实验表格
3. 扩充参考文献到30+篇
4. 确保tex可编译
5. Ablation study表格

### 第八位 Claude (M176-M200) — 待执行
**角色**: 消融实验 + PEMS-BAY

1. 逐个关闭: SE / cascade residual / depth gate / freq injection
2. 每个跑完push结果
3. PEMS-BAY数据集实验

---

## 实验运行指南

```bash
# 在 ags1 上
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git pull origin main

# 标准实验 (H100, 200 epoch)
GPU=2 EPOCHS=200 nohup bash experiments/run_server_experiment.sh &

# 多种子
for SEED in 42 123 456; do
  SEED=$SEED GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
done

# 查看结果
cat experiments/results/summary.json
```

## 子Claude派发 (通过claude_hk_chat.sh)

```bash
# 同步cookie
cd .claude-hk-config && git pull

# 派发任务
bash claude_hk_chat.sh "你是第N位Claude(M_start-M_end)。
clone github.com/dylanyunlon/walpurgis-WTFGG, 查看结构, 
执行任务: [具体任务描述]
铁律: 不开分支、不用后缀、改算法不改字符串。
Token: <GIT_TOKEN_FROM_ENV>"
```

## 关键文件索引

| 文件 | 用途 |
|------|------|
| src/walpurgis/models/model.py | 主模型 (SE+cascade+depth gate+freq injection) |
| src/walpurgis/models/losses.py | 损失函数 (cascade_aware + adaptive LogCosh) |
| src/walpurgis/models/trainer.py | 训练循环 (CL sigmoid + diagnostics) |
| src/walpurgis/models/diffusion_block/dif_model.py | 扩散图卷积 (chunked gconv) |
| src/walpurgis/configs/METR-LA.yaml | 超参数 |
| train_walpurgis.py | 训练入口 |
| experiments/run_server_experiment.sh | 服务器全流程实验 |
| claude_hk_chat.sh | 子Claude派发 |
| bench/sota.json | SOTA 对比 |
| walpurgis_reconstructed.tex | 论文 |
