# Walpurgis Multi-Claude Dispatch Plan — SOTA Experiment Phase

## 目标
在 METR-LA 和 PEMS-BAY 数据集上运行实验，获得超越 STAEFormer (SOTA) 的结果。
当前SOTA: STAEFormer(2023) MAE=2.90, RMSE=5.91, MAPE=8.12% (METR-LA)

## SOTA Baselines (METR-LA)
| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| PDFormer | 2023 | 2.94 | 6.08 | 8.56% |
| D2STGNN | 2022 | 3.04 | 6.23 | 8.33% |
| STG-NCDE | 2022 | 2.96 | 6.51 | 9.13% |
| AGCRN | 2020 | 3.07 | 6.34 | 9.81% |

## 关键规则
1. **算法改写，不改字符串/docstring** — 每个改动必须改变数学计算逻辑
2. **直接push到main** — 不开新分支，不加v2/v3/port后缀
3. **复用conda环境** `walpurgis` — 如有则直接activate
4. **Token**: ${GITHUB_TOKEN}
5. **Claude截断时发送Continue继续**

## Claude分工

### 第一位Claude (已完成) — M074-M078: cathexis变体创建
- [x] M074: 创建walpurgis_cathexis 10项算法改写
- [x] M075: SYNTH smoke test通过
- [x] M076: 创建server_setup.sh, run_experiment.sh
- [x] M077: 创建dispatch计划和子Claude prompt
- [x] M078: push到main

### 第二位Claude — M079-M083: 服务器环境 + METR-LA数据
任务: 在GPU服务器上配置环境，下载METR-LA数据集，运行cathexis实验
- M079: 拉取dylanyunlon/claude-hk-config同步cookie
- M080: 执行server_setup.sh配置conda+GPU环境
- M081: 下载METR-LA数据集（DCRNN格式: train/val/test.npz + adj_mx_la.pkl）
- M082: 运行 ./run_experiment.sh cathexis METR-LA cuda:0 80
- M083: push结果到git, 更新comparison_table.tex中cathexis行

### 第三位Claude — M084-M088: PEMS-BAY + 多变体对比
任务: 下载PEMS-BAY数据，运行cathexis + corona + zenith三变体
- M084: 下载PEMS-BAY数据集
- M085: ./run_experiment.sh cathexis PEMS-BAY cuda:0 80
- M086: ./run_experiment.sh corona METR-LA cuda:0 80 (对比)
- M087: ./run_experiment.sh zenith METR-LA cuda:0 80 (对比)
- M088: 汇总三变体结果到comparison_table.tex

### 第四位Claude — M089-M093: 超参数调优
任务: 基于第二/三位Claude的初始结果，进行超参调优
- M089: 分析cathexis在METR-LA的初始结果
- M090: 调整lr/batch_size/hidden_dim/dropout
- M091: 重新训练最优配置 (3 seeds)
- M092: 运行10维评估 (bench/walpurgis_eval.py)
- M093: push最终结果

### 第五位Claude — M094-M098: tex表格填充 + upstream baseline验证
任务: 运行upstream D2STGNN作为基准对比, 填充完整tex表格
- M094: 运行upstream/d2stgnn在METR-LA (验证我们的baseline数字)
- M095: 运行upstream/d2stgnn在PEMS-BAY
- M096: 收集所有变体结果，生成完整comparison_table.tex
- M097: 更新walpurgis_reconstructed.tex Section 5 (Evaluation)
- M098: push最终论文就绪的tex

### 第六位Claude — M099-M103: 消融实验 + 最终验证
任务: 消融实验验证每个算法改写的贡献
- M099: 逐一禁用cathexis的10个改写，运行消融
- M100: 生成ablation_table.tex
- M101: 运行D7 (Graph Sensitivity) 和 D10 (Stability) 评估
- M102: 最终清理仓库 (移除临时文件)
- M103: 最终论文数据验证 + push

## 数据下载指南

### METR-LA
来源: DCRNN preprocessing (Li et al., ICLR 2018)
```bash
# 方案1: 从zenodo
wget https://zenodo.org/record/5724362/files/METR-LA.zip

# 方案2: 从LibCity项目
git clone https://github.com/LibCity/Bigscity-LibCity-Datasets.git
# 找到METR-LA

# 方案3: 手动（若以上失败）
# 原始csv: https://github.com/liyaguang/DCRNN/tree/master/data
# 用DCRNN的generate_training_data.py预处理
```

文件结构:
```
datasets/METR-LA/
├── train.npz  (x, y, x_offsets, y_offsets)
├── val.npz
└── test.npz
datasets/sensor_graph/
└── adj_mx_la.pkl  (距离邻接矩阵, 207 nodes)
```

### PEMS-BAY
同样的预处理流程，325 nodes:
```
datasets/PEMS-BAY/
├── train.npz, val.npz, test.npz
datasets/sensor_graph/
└── adj_mx_bay.pkl
```
