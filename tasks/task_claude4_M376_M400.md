在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第四位Claude(M376-M400)，Phase 2的消融实验执行者。

你的任务: 逐个关闭新算法组件，跑消融实验
1. 消融1: 关闭自适应时空嵌入 (设 adp_gate 初始=0.0 并 freeze)
2. 消融2: 关闭PSD不确定性加权 (_use_uncertainty=False)
3. 消融3: 关闭输出头时序注意力 (bypass temporal_cross_attn)
4. 消融4: 关闭SE通道注意力
5. 消融5: 关闭cascade残差聚合
6. 每个配置在SYNTH上跑5 epoch，记录MAE变化
7. 结果写入 experiments/results/ablation.json
8. 填入 walpurgis_reconstructed.tex 的消融表

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
