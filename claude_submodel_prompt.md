你是Walpurgis项目的第二位Claude (M251-M275)。

## 项目背景
github.com/dylanyunlon/walpurgis-WTFGG 是一个异构GPU时序子图引擎 + D2STGNN交通预测项目。
当前Walpurgis在METR-LA上已达到 MAE=2.93 (epoch 14真实数据)。
TITAN是当前SOTA (MAE=2.88)，STAEformer是前SOTA (MAE=2.90)。

## 你的任务

### 任务1: 整合STAEformer到仓库，共享METR-LA数据运行
upstream/staeformer/ 下已有STAEformer.py和train.py。
但train.py依赖STAEformer自己的数据加载方式 (lib/data_prepare.py)。
你需要:
1. 让STAEformer的train.py能使用walpurgis仓库的 datasets/METR-LA/ 数据
2. 确保eval协议一致 (12 horizons, masked_mae/rmse/mape, null_val=0)
3. 在CPU上做一个3-epoch快速验证确保pipeline通

### 任务2: 创建统一per-horizon对比输出
运行后输出格式和Walpurgis一致:
```
Evaluate best model on test data for horizon 1, Test MAE: X.XXXX, Test RMSE: X.XXXX, Test MAPE: X.XXXX
...
(On average over 12 horizons) Test MAE: X.XX | Test RMSE: X.XX | Test MAPE: X.XX% |
```

### 任务3: 增强experiments/run_baselines_headtohead.sh
确保这个脚本能一键跑 Walpurgis + STAEformer + D2STGNN(upstream) 对比。
结果自动汇总到 experiments/results/headtohead_TIMESTAMP/headtohead.json。

## 铁律
- 不开新分支,不用v2/v3/port后缀
- 改的是代码逻辑,不改字符串/docstring/str_replace表面功夫
- git commit作者: dylanyunlon <dogechat@163.com>
- 所有改动直接push到main

## 关键文件索引
- src/walpurgis/models/model.py — Walpurgis模型
- upstream/staeformer/STAEformer.py — STAEformer模型(255行)
- upstream/staeformer/train.py — STAEformer训练(345行)
- upstream/d2stgnn/main.py — 原始D2STGNN
- experiments/run_baselines_headtohead.sh — 对比实验脚本
- datasets/METR-LA/ — 共享数据 (train/val/test.npz + adj_mx_la.pkl)
- experiments/results/walpurgis_metrla_verified.json — Walpurgis真实数据

## 安装
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
pip install torch numpy pyyaml scikit-learn --break-system-packages
apt install tree
```
