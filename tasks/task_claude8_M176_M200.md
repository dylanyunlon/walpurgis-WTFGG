不要立刻查看所有内容,在你的linux上用tree、git branch先查看架构。使用clone工具进行git clone。没有tree你就apt install tree。

github.com/dylanyunlon/walpurgis-WTFGG

你是第八位Claude(M176-M200)。你的角色是消融实验 + PEMS-BAY数据集。

查看 MULTI_CLAUDE_PLAN.md 了解完整进度。

你的任务:
1. 设计消融实验脚本 (experiments/run_ablation.sh):
   - 关闭SE: 在model.py的embed_se和se_block处bypass
   - 关闭cascade residual: 在model.py的cascade_aggregate处归零
   - 关闭depth gate: 固定所有gate=1.0
   - 关闭freq injection: 在model.py的freq_gate处归零
   - 每个消融配置独立跑一轮, 结果push到 experiments/results/ablation_*
2. PEMS-BAY数据集:
   - 检查 datasets/PEMS-BAY/ 或 src/walpurgis/configs/PEMS-BAY.yaml
   - 跑PEMS-BAY实验

铁律:
- 不开新分支，改的是算法
- 消融通过代码中的开关控制(config参数)，不是删代码
- 作者: dylanyunlon <dogechat@163.com>
