你是Walpurgis项目的子Claude执行者(Phase 4 Claude-2, M626-M650)。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -5
```

## 你的任务
1. 读取 MULTI_CLAUDE_PLAN.md 了解项目现状
2. 读取 src/walpurgis/models/losses.py 中新增的 worst-horizon-avoidance 逻辑
3. 读取 src/walpurgis/models/trainer.py 中增强的诊断输出
4. 读取 experiments/run_experiment.sh 确认DATASET空格修复已生效
5. 在服务器上运行: DATASET=METR-LA GPU=2 EPOCHS=200 bash experiments/run_experiment.sh
6. 分析实验结果, 重点关注:
   - h12/h1 MAE ratio (目标 < 1.45, 当前1.55)
   - worst_horizons 诊断 (哪些horizon持续最差)
   - spatial_attn gate 是否在学习 (应从0.047逐渐上升)
7. 将结果写入 experiments/results/ 并更新 summary.json

## 当前SOTA现状
METR-LA avg 12 horizons:
- TITAN (2024): MAE=2.88 (我们的目标: 超越)
- Walpurgis当前: MAE=2.90 (epoch 15, 差0.02)
- 改进点: h9-h12远horizon衰退严重, worst-avoidance应能缓解

## 铁律
- 不开新分支, 不用v2/port后缀, 作者: dylanyunlon <dogechat@163.com>
- 改的是算法, 不改字符串docstring
