# Walpurgis Multi-Claude 开发计划

## 状态总结 (Claude-1 完成)

### 当前问题
1. **实验结果远未SOTA**: 当前最佳 MAE=3.08 (Epoch 5), 目标 <2.85, SOTA=2.88 (TITAN)
2. **训练发散**: Curriculum learning 启动后(Epoch 6+) MAE 飙升到 4-6+, Early Stop 在 Epoch 20
3. **配置被覆盖**: YAML 写 num_hidden=64 但日志显示实际用了 num_hidden=32 — 模型太小
4. **结果未回写**: summary.json 全是 N/A
5. **C++ 引擎和 Python 训练完全解耦**: 没有集成
6. **TeX 参考文献太少**: 只有 9 篇, 需要 30+

### 服务器信息
- **ags1**: 2x A6000 (48GB) + 1x H100 NVL (96GB), AMD EPYC 9354 128核, ~1.5TB RAM
- **CUDA**: 11.5, Driver 550.144, Ubuntu 22.04
- **Conda**: walking3 环境
- **GPU拓扑**: 全在 NUMA node1, 无NVLink

### SOTA 对比 (METR-LA, avg 12 horizons)
| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| TITAN | 2024 | 2.88 | 5.33 | — |
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| PDFormer | 2023 | 2.94 | 6.08 | 8.56% |
| STEP | 2022 | 2.98 | — | — |
| D2STGNN (upstream) | 2022 | 3.04 | 6.23 | 8.33% |
| **Walpurgis (当前)** | — | **3.08** | **6.04** | **8.17%** |
| 目标 | — | **<2.85** | — | — |

### 仓库凭证
- URL: https://github.com/dylanyunlon/walpurgis-WTFGG.git
- Token: 由项目管理员通过 prompt 提供, 不要提交到仓库

### 铁律
- **不开新分支**, 不用 v2/v3/port 等后缀
- **不改字符串/docstring/str_replace** 这种表面功夫 — 改的是算法
- 所有改动直接 push 到 main
- 实验结果自动写入 experiments/results/ 并 push

---

## Claude 任务分配

### Claude-1 (M001-M025) — 已完成
**角色**: 项目审计 + 基础设施
- 分析项目结构和所有文件
- 诊断训练发散原因
- 更新 SOTA 对比表 (bench/sota.json)
- 创建自动化实验脚本 (experiments/auto_experiment.sh)
- 制定本计划

### Claude-2 (M026-M050) — 算法核心修复
**角色**: 修复训练发散 + 模型扩容
**优先级最高 — 不修复这些后面全白做**

关键任务:
1. **修复配置覆盖bug**: 找到 num_hidden 被从 64 覆盖为 32 的位置, 确保 YAML 配置被正确使用
   - 检查 train_walpurgis.py 的 run() 函数和 set_config() 调用链
   - 检查 src/walpurgis/utils/train.py 中的 set_config 和 load_model
   - 确认 model_args 没有被 checkpoint 或 hardcoded 值覆盖

2. **修复 Curriculum Learning 发散**: Epoch 6 启动 CL 后立刻崩溃
   - 检查 src/walpurgis/models/trainer.py 中的 cl_steps / warm_steps 计算
   - CL 的学习率重置 "reset the learning rate to 0.002" 可能太激进
   - 考虑: CL 渐进比例 sigmoid ramp, 而不是硬切换
   - 检查 src/walpurgis/models/losses.py 中的 cascade-aware loss 在 CL 模式下的行为

3. **模型扩容到充分利用 H100 (96GB)**:
   - num_hidden: 64→128, node_hidden: 32→64, time_emb_dim: 32→64
   - batch_size: 64→128 (H100 够用)
   - 添加 gradient accumulation 选项

4. **在服务器上运行修复后的实验**:
   ```
   cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
   git pull origin main
   GPU=2 EPOCHS=200 bash experiments/auto_experiment.sh
   ```

验收标准: 训练不再在 Epoch 6 发散, 能跑完 100+ epochs, Val MAE < 3.0

### Claude-3 (M051-M075) — 算法改进 Phase 1
**角色**: D2STGNN → Walpurgis 算法升级

基于 output/variant_analysis.json 中排名前3的变体:

1. **SE Channel Attention** (cascade 变体, rank 1):
   - 已有代码在 src/walpurgis/models/model.py 的 SqueezeExcitation 类
   - 验证它是否真的被用在 forward pass 中
   - 确保 reduction ratio 合理

2. **Dense Cascade Residual**: 每层 backcast → 最终聚合的跳连

3. **Dynamic Depth Gating**: depth_gates sigmoid gate 初始化应偏向 1.0

4. **Horizon-weighted Loss**: 远horizon权重更高

每做一个改动就跑一轮实验, 用 auto_experiment.sh push 结果.

验收标准: MAE 降到 3.0 以下

### Claude-4 (M076-M100) — 算法改进 Phase 2
**角色**: 冲击 SOTA

1. **学习率策略优化**: CosineAnnealingWarmRestarts 或 OneCycleLR
2. **图结构增强**: adaptive adj learning
3. **数据增强**: Time-shift, Node dropout, Mixup
4. **多种子评估**: seed=42,123,456

验收标准: MAE < 2.90

### Claude-5 (M101-M125) — TeX 论文 + 实验数据填充
**角色**: 论文完善

1. **填充实验数据到 tex**: 从 experiments/results/ 自动提取
2. **扩充参考文献到 30+ 篇**
3. **确保 tex 可编译**
4. **Ablation study 表格**

### Claude-6 (M126-M150) — C++ 引擎集成 + 最终优化
**角色**: 系统集成 + 收尾

1. **C++ benchmark 数据**: make cpu / make cuda
2. **PEMS-BAY 实验**
3. **最终清理 + README 更新**

---

## 实验运行指南

```bash
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git pull origin main
GPU=2 EPOCHS=200 bash experiments/auto_experiment.sh
```

## 关键文件索引

| 文件 | 用途 |
|------|------|
| src/walpurgis/models/model.py | 主模型 |
| src/walpurgis/models/losses.py | 损失函数 |
| src/walpurgis/models/trainer.py | 训练循环 + CL + LR |
| src/walpurgis/configs/METR-LA.yaml | 超参数 |
| train_walpurgis.py | 训练入口 |
| experiments/auto_experiment.sh | 自动实验 |
| bench/sota.json | SOTA 对比 |
| walpurgis_reconstructed.tex | 论文 |
