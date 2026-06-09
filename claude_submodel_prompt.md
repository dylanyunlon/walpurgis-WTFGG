在你的linux上执行以下步骤。不要立刻查看所有内容。

apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -5
cat MULTI_CLAUDE_PLAN.md

你是第八位Claude(M176-M200)。查看MULTI_CLAUDE_PLAN.md了解完整进度。

你的任务:
1. 查看 experiments/results/ 下最新result.json和summary.json获取实验数据
2. 查看 src/walpurgis/configs/METR-LA.yaml 和 src/walpurgis/models/ 下的算法文件
3. 针对当前MAE=2.90 → 目标<2.85，做以下算法调整(改算法!):
   - 在 src/walpurgis/models/model.py 的输出头增加dropout(0.1)防止过拟合
   - 在 src/walpurgis/configs/METR-LA.yaml 中将num_hidden从96调到112
   - 调整losses.py中horizon_scale从0.12到0.15以更强调远horizon
4. 添加SYNTH数据集快速验证: python3 train_walpurgis.py --dataset SYNTH --epochs 5 --debug
5. 将改动commit并push到main

铁律:
- 不开新分支，不用v2/v3/port等后缀
- 改的是算法，不改字符串/docstring
- 作者: dylanyunlon <dogechat@163.com>
- git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git

如果输出被截断，请发送Continue继续执行。
