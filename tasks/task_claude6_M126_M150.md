不要立刻查看所有内容,在你的linux上用tree、git branch先查看架构。使用clone工具进行git clone。没有tree你就apt install tree。

github.com/dylanyunlon/walpurgis-WTFGG

你是第六位Claude(M126-M150)。前五位Claude已经完成了项目审计、训练修复、算法改进(SE+cascade+depth gate)、SOTA冲刺(CosineAnnealing+adaptive adj)、和本轮的内存优化+断点调试。

查看 MULTI_CLAUDE_PLAN.md 了解完整进度。查看 experiments/results/ 下最新的 result.json 和 summary.json 获取实验数据。

你的任务:
1. 拉取最新代码，查看实验结果
2. 分析per-horizon MAE分布，找到薄弱的horizon
3. 针对性算法调整:
   - 如果远horizon(9-12)MAE高: 加大losses.py中horizon_scale (0.08→0.12)
   - 如果训练早期震荡: 调整trainer.py中CL warm_epochs
   - 如果整体偏高: 尝试增加num_hidden (64→96)，分块gconv已解决内存问题
4. 在服务器跑实验: GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
5. 多种子评估: seed=42,123,456

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 所有改动直接commit到main
- 作者: dylanyunlon <dogechat@163.com>

服务器信息:
- ags1: 2x A6000 (48GB) + 1x H100 NVL (96GB)
- Conda: walking3 环境
- 项目路径: /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG

Git token通过环境变量 GIT_TOKEN 传递。
