在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第二位Claude(M326-M350)，Phase 2的服务器实验执行者。

你的任务: 在服务器环境跑METR-LA完整实验，拿到SOTA数据
1. 查看 src/walpurgis/configs/METR-LA.yaml 确认 num_hidden=112
2. 查看最新的算法改动: git diff HEAD~1 -- src/walpurgis/models/
3. 用SYNTH快速验证pipeline: python3 train_walpurgis.py --dataset SYNTH --epochs 3 --debug
4. 如果跑通，设置METR-LA实验:
   GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
5. 跑完后提取结果到 experiments/results/summary.json
6. 如果 MAE < 2.85: 写入 walpurgis_reconstructed.tex 的SOTA表格
7. 多种子评估: SEED=42,123,456 各跑一轮

关键新算法（本轮Claude-1 M301-M325已完成）:
- 自适应时空嵌入 (adaptive spatio-temporal embedding from STAEformer)
- PSD不确定性感知损失加权 (uncertainty-aware loss from TITAN)
- 输出头时序自注意力 (temporal cross-attention in output head)

诊断重点:
- 观察 [DIAG] 日志中 adaptive_emb_gate 的变化趋势
- 观察 uncertainty_mean 是否在epoch 20后生效
- 关注 per_horizon_MAE 的长horizon（h9-h12）是否改善

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
- git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git
