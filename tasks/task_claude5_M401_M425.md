在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第五位Claude(M401-M425)，Phase 2的多种子+超参搜索。

你的任务:
1. 三种子评估: SEED=42,123,456 在METR-LA上跑完整实验
2. 如果MAE仍 > 2.85:
   - 尝试 num_hidden=128 (更大容量)
   - 尝试 horizon_scale=0.20 (更强远horizon权重)
   - 尝试 dropout=0.15 (更强正则)
3. 记录每次结果到 experiments/results/
4. 最佳结果更新 bench/sota.json

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
