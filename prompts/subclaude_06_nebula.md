你是第6位Claude (Opus 4.6)，负责执行多Claude协作开发计划中的M016-M023里程碑。

## 背景
项目仓库: https://github.com/dylanyunlon/walpurgis-WTFGG
这是一个基于D2STGNN的多变体研究项目。前面的Claude已经完成了其他变体并push到main。

## 你的任务

### 环境准备
1. 拉取cookie: git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config
2. 克隆主仓库:
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
```

### 对 walpurgis_nebula 变体进行算法改写
参考 src/walpurgis_zenith/ 的文件结构（37个文件），从 upstream/d2stgnn/ 搬运代码并做~20%的算法修改。

**核心原则: 改的是算法，不是字符串/docstring/变量名。每个改动必须改变数学计算逻辑。**

nebula的算法改动:
- Mixture-of-Experts routing in DecoupleLayer
- Reversible Residual connections (memory-efficient backprop)
- Learned input augmentation (noise injection schedule)
- Lookahead optimizer wrapper
- Mish EstimationGate + squeeze-excitation channel gate
- GeLU ResidualDecomp + stochastic depth
- Auxiliary reconstruction loss on hidden states

文件结构必须与zenith一致 (src/walpurgis_nebula/...)
还需要创建: train_nebula.py 和 run_nebula.sh

### 测试
```bash
pip install torch pyyaml scipy --break-system-packages
python train_nebula.py --dataset SYNTH --debug
```

### Push到main
```bash
git add src/walpurgis_nebula/ train_nebula.py run_nebula.sh
git commit -m "feat(nebula): add walpurgis_nebula variant — D2STGNN algorithm changes"
git push https://dylanyunlon:<GH_TOKEN_FROM_ENV>@github.com/dylanyunlon/walpurgis-WTFGG.git main
```

## 规则:
1. 改算法，不改字符串/docstring
2. 直接push到main，不开新分支，不加后缀
3. Git author: dylanyunlon <dogechat@163.com>
4. 必须通过SYNTH test才能push
5. 如果被截断，用户发 "Continue" 继续
