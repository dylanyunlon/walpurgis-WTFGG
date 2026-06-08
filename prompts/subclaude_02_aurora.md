你是第二位Claude (Opus 4.6)，负责执行多Claude协作开发计划中的M004-M007里程碑。

## 背景
项目仓库: https://github.com/dylanyunlon/walpurgis-WTFGG
这是一个基于D2STGNN（解耦时空图神经网络）的多变体研究项目。第一位Claude已经完成了walpurgis_zenith变体的创建并push到了main分支。

## 你的任务: M004-M007

### M004: 环境准备
1. 先拉取cookie配置:
```bash
git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config
```
2. 克隆主仓库:
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
```

### M005: 对 walpurgis_aurora 变体进行算法改写
参考 src/walpurgis_zenith/ 的文件结构（37个文件），从 upstream/d2stgnn/ 搬运代码并做~20%的算法修改。

**核心原则: 改的是算法，不是字符串/docstring/变量名。每个改动必须改变数学计算逻辑。**

aurora的算法改动方向:
1. **Multi-Scale Temporal Attention**: 在InhBlock中用多尺度时间注意力替代单一GRU序列处理。使用不同的kernel size做temporal pooling后做cross-scale attention
2. **Gated Fusion**: 在DecoupleLayer中，用门控机制(sigmoid gate)替代简单的残差加法来融合diffusion和inherent分支的输出
3. **自适应图正则化**: 在DynamicGraphConstructor中添加spectral graph regularizer，约束学到的图的拉普拉斯特征值分布
4. **Cosine Annealing with Warm Restarts**: 替代MultiStepLR调度器
5. **Huber Loss**: 在loss函数中用Huber Loss替代纯MAE，增加outlier鲁棒性
6. **EstimationGate**: 用GELU替代ReLU，增加可学习的scale因子
7. **ResidualDecomp**: 用SiLU替代ReLU，增加可学习的momentum参数

文件结构必须与zenith一致:
```
src/walpurgis_aurora/
├── __init__.py          (诊断工具)
├── main.py
├── generate_synth_data.py
├── configs/SYNTH.yaml
├── dataloader/
├── datasets/
├── utils/{__init__,cal_adj,load_data,log,train}.py
├── models/
│   ├── __init__.py, model.py, trainer.py, losses.py
│   ├── decouple/{estimation_gate,residual_decomp}.py
│   ├── diffusion_block/{__init__,dif_block,dif_model,forecast}.py
│   ├── inherent_block/{__init__,inh_block,inh_model,forecast}.py
│   └── dynamic_graph_conv/{__init__,dy_graph_conv}.py
│       └── utils/{__init__,distance,mask,normalizer}.py
```

还需要在根目录创建:
- train_aurora.py
- run_aurora.sh

### M006: 测试
```bash
pip install torch pyyaml scipy --break-system-packages
python train_aurora.py --dataset SYNTH --debug
```
必须成功跑通3个epoch。

### M007: Push到main
```bash
git add src/walpurgis_aurora/ train_aurora.py run_aurora.sh
git commit -m "feat(aurora): add walpurgis_aurora variant — D2STGNN algorithm changes

算法修改:
- Multi-Scale Temporal Attention (替代单一GRU)
- Gated Fusion (sigmoid gate融合diffusion/inherent)
- Spectral graph regularizer
- CosineAnnealingWarmRestarts
- Huber Loss
- GELU EstimationGate + learnable scale
- SiLU ResidualDecomp + momentum

SYNTH smoke test passed"

git push https://dylanyunlon:<GH_TOKEN_FROM_ENV>@github.com/dylanyunlon/walpurgis-WTFGG.git main
```

## 关键规则:
1. **改算法，不改字符串/docstring/变量名** — 每个改动必须改变数学计算
2. **直接push到main** — 不开新分支，不加v2/v3/port后缀
3. **Git author**: dylanyunlon <dogechat@163.com>
4. **必须通过SYNTH test** 才能push
5. 如果被截断了，用户会发送 "Continue" 让你继续

开始执行吧。先M004环境准备，然后M005写所有文件，M006跑测试，M007 push。
