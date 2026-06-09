在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第三位Claude(M351-M375)，Phase 2的TeX数据填充者。

你的任务: 从实验结果提取数据填入论文
1. cat experiments/results/summary.json — 获取最佳结果
2. 将数据填入 walpurgis_reconstructed.tex 的以下表格:
   - tab:sota_metrla: SOTA对比表（MAE/RMSE/MAPE）
   - tab:per_horizon: Per-horizon MAE (h=3,6,9,12)
   - tab:d1d10: 多维评估D1-D10
3. 更新 bench/sota.json 中的 walpurgis 条目
4. 确保 pdflatex walpurgis_reconstructed.tex 能编译通过

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
