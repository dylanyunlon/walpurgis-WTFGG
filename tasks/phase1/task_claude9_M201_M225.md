在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第九位Claude(M201-M225)。查看MULTI_CLAUDE_PLAN.md了解完整进度。

你的任务: 在服务器环境验证实验能跑通
1. 查看 src/walpurgis/configs/METR-LA.yaml 确认 num_hidden=112
2. 查看 experiments/results/summary.json 获取当前数据
3. 用SYNTH数据集验证训练pipeline能跑: python3 train_walpurgis.py --dataset SYNTH --epochs 3 --debug
4. 如果跑通, commit验证日志并push

注意: 服务器上的正式METR-LA实验由用户手动执行:
  GPU=2 EPOCHS=200 GIT_TOKEN=$GIT_TOKEN bash experiments/run_server_experiment.sh

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
- git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git
