# 任务: walpurgis_eclipse 变体 — 第四位Claude (Opus 4.6)

你是第四位Claude (Opus 4.6)，接力开发 walpurgis-WTFGG 项目。

## 里程碑: M700-M724

## 项目上下文
仓库: https://github.com/dylanyunlon/walpurgis-WTFGG.git
GitHub Token: <GITHUB_TOKEN>
Git作者: dylanyunlon <dogechat@163.com>

这是一个D2STGNN时空图神经网络的多变体移植项目。upstream/d2stgnn/ 是原始代码。
已有变体: walpurgis (基础), walpurgis_nightfall, walpurgis_cardgame, walpurgis_walking
你需要创建新变体: walpurgis_eclipse

## 关键规则 (必须遵守)
1. **不开新分支** — 直接在main上工作
2. **不加v10/port/v2/v3等后缀** — 目录名就是 src/walpurgis_eclipse/
3. **改的是算法** — 不是改字符串/docstring/注释/变量名; 每个核心.py文件至少20%实质性算法改动
4. **添加debug断点调试** — 每个模块都要print当前数据状态,让运行时能看到反馈
5. **git push直接到main** — 作者: dylanyunlon <dogechat@163.com>

## 第一步: 克隆+环境

```bash
apt-get update -qq && apt-get install -y -qq tree git
pip install torch numpy pyyaml scikit-learn --break-system-packages -q
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
git remote set-url origin https://dylanyunlon:<GITHUB_TOKEN>@github.com/dylanyunlon/walpurgis-WTFGG.git
tree upstream/d2stgnn --charset ascii
```

## 第二步: 创建 src/walpurgis_eclipse/ — 完整移植所有upstream文件

从 upstream/d2stgnn/ 移植所有文件到 src/walpurgis_eclipse/，以下是每个文件的算法改动要求:

### 核心模型 (models/)

**__init__.py**: 导出 trainer
```python
from .trainer import trainer
```

**model.py** — D2STGNN主模型:
- upstream用 `sum()` 聚合层输出 → eclipse改为**指数移动平均(EMA)聚合**: 维护可学习衰减率λ, 后面层的权重按λ^(L-l)指数衰减
- upstream用 `F.relu` 输出头 → eclipse改为 **GELU + SpectralNorm** 双层输出
- upstream embedding后直接进layer → eclipse在embedding后加 **Gaussian noise注入** (训练时σ=0.01, eval时关闭)
- 添加ECLIPSE_DEBUG: 打印每层DecoupleLayer的输入/输出shape, dif/inh forecast能量比

**trainer.py** — 训练引擎:
- upstream用 Adam → eclipse改为 **AdamW + ReduceLROnPlateau** (patience=5, factor=0.5)
- upstream固定clip=5 → eclipse改为 **自适应梯度裁剪**: 按p95分位数动态调整clip阈值
- upstream curriculum learning按步数线性增长 → eclipse改为 **对数增长**: cl_len = 1 + int(log2(1 + batch_num/cl_steps) * output_seq_len)
- 添加debug: 每个train step打印loss, grad_norm, clip_threshold, lr, cl_len

**losses.py** — 损失函数:
- upstream用 masked_mae (L1) → eclipse改为 **Tukey biweight loss**: ρ(r) = (c²/6)[1-(1-(r/c)²)³] 当|r|≤c, c²/6 当|r|>c, c=4.685
- 新增 **gradient_penalty**: 惩罚预测值在时间维度上的二阶差分过大, α * mean(|d²pred/dt²|)
- 保留masked_rmse/masked_mape作为评估指标
- 添加debug: 打印loss各component的值

### 解耦模块 (models/decouple/)

**estimation_gate.py**:
- upstream用 2层FC+ReLU → eclipse改为 **3层FC + Swish激活 + ChannelAttention**: 先FC降维→Swish→FC→ChannelSE(squeeze-excitation)→FC→sigmoid
- 可学习温度参数τ控制sigmoid锐度: sigmoid(x/τ)
- debug: 打印gate值分布 (mean/std/min/max)

**residual_decomp.py**:
- upstream用 LayerNorm(x - ReLU(y)) → eclipse改为 **RMSNorm(x - Mish(y) * α)**, α是可学习标量, Mish(x) = x * tanh(softplus(x))
- debug: 打印残差范数

### 扩散块 (models/diffusion_block/)

**__init__.py**: 从 dif_block 导出 DifBlock

**dif_model.py** — STLocalizedConv:
- upstream用 BN+ReLU → eclipse改为 **InstanceNorm2d + GELU**
- upstream gconv只有线性聚合 → eclipse在gconv里加 **残差skip连接**: out = gcn_updt(cat) + X_0
- debug: 打印gconv前后的tensor统计

**dif_block.py** — DifBlock:
- upstream backcast是单层Linear → eclipse改为 **2层MLP + LayerNorm + sigmoid门控**: gate = sigmoid(MLP(h)), backcast = gate * MLP(h)
- 在forecast_branch前加 **Dropout(p=0.1)**
- debug: 打印backcast vs forecast的能量比

**forecast.py** — Diffusion forecast:
- upstream直接用history padding → eclipse改为 **反射padding (reflect)**: 不够时用序列反转填充
- forecast_fc后加 **layer_norm**
- debug: 打印predict序列长度和值范围

### 动态图卷积 (models/dynamic_graph_conv/)

**__init__.py**: 导出 DynamicGraphConstructor

**dy_graph_conv.py**:
- 保持整体结构不变
- debug: 打印dynamic graph的稀疏度和值范围

**utils/__init__.py**: 导出 DistanceFunction, Mask, Normalizer, MultiOrder

**utils/distance.py**:
- upstream用 scaled-dot attention → eclipse改为 **cosine similarity + 可学习偏置**: sim = cos(Q,K) * τ + bias, τ可学习
- upstream BN → eclipse改为 **LayerNorm**
- debug: 打印attention score分布

**utils/mask.py**:
- upstream硬mask → eclipse改为 **soft gating**: mask = sigmoid(learnable_logits) * adj, logits是nn.Parameter
- debug: 打印mask的稀疏度

**utils/normalizer.py**:
- upstream用 D⁻¹A 行归一化 → eclipse改为 **D⁻½AD⁻½ 对称归一化**
- 保持MultiOrder不变
- debug: 打印归一化后的行和

### 内生块 (models/inherent_block/)

**__init__.py**: 导出 InhBlock

**inh_model.py** — RNNLayer + TransformerLayer:
- RNNLayer: GRU后接 **LayerNorm** 稳定隐状态
- TransformerLayer: 加 **残差连接**: output = X + attention(X,K,V)
- debug: 打印GRU hidden state统计, attention score分布

**inh_block.py** — InhBlock:
- upstream backcast是单层FC → eclipse改为 **sigmoid门控残差**: gate = sigmoid(FC(h)), backcast = gate * FC(h) + (1-gate) * h
- PositionalEncoding: 加 **可学习phase offset** — 每个频率分量的相位可微调
- debug: 打印inherent model各阶段tensor shape

**forecast.py** — Inherent forecast:
- upstream直接cat predict → eclipse加 **步长衰减权重**: 越远的predict步权重越小, weight = exp(-decay * step)
- debug: 打印forecast step和weight

### 工具 (utils/)

**__init__.py**: 空

**cal_adj.py**:
- 保持所有邻接矩阵计算函数不变
- 改写 check_nan_inf: 增加 **详细位置报告** (哪些元素是nan/inf)
- debug: 打印adj矩阵统计

**load_data.py**:
- StandardScaler: 加 **eps-guarded除法** (除以max(std, 1e-8))
- load_dataset: 加 **数据完整性检查** (NaN检测, shape验证)
- debug: 打印数据加载统计

**log.py**:
- upstream只复制文件做log → eclipse改为 **JSONL结构化日志** + CSV metrics dual dump
- TrainLogger: 每个epoch的metrics写入 events.jsonl 和 metrics.csv
- debug: 打印日志路径

**train.py**:
- set_config: 用 **Knuth乘法散列** 派生子seed
- EarlyStopping: 加 **趋势检测** — 用最近8个epoch的线性回归斜率判断是否收敛
- data_reshaper: 加 **dtype检查** — float64自动转float32, NaN替换为0
- debug: 打印seed/patience/趋势斜率

### 数据加载器 (dataloader/)

**__init__.py**: 导出 DataLoader

**dataloader.py**:
- upstream用 np.random.permutation shuffle → eclipse改为 **Fisher-Yates (Knuth) in-place shuffle**
- upstream用last_sample padding → eclipse改为 **环形wrap padding**: 用序列开头的样本填充
- debug: 打印batch数量和padding数量

### 数据集 (datasets/)

创建以下文件:
- datasets/__init__.py
- datasets/raw_data/_gen_speed_common.py — METR-LA/BAY共用生成逻辑
- datasets/raw_data/_gen_flow_common.py — PEMS04/08共用生成逻辑  
- datasets/raw_data/_gen_adj_common.py — adj生成共用逻辑
- datasets/raw_data/METR-LA/generate_training_data.py — 调用_gen_speed_common
- datasets/raw_data/PEMS-BAY/generate_training_data.py — 调用_gen_speed_common
- datasets/raw_data/PEMS04/generate_training_data.py — 调用_gen_flow_common
- datasets/raw_data/PEMS04/generate_adj_mx.py — 调用_gen_adj_common
- datasets/raw_data/PEMS08/generate_training_data.py — 调用_gen_flow_common
- datasets/raw_data/PEMS08/generate_adj_mx.py — 调用_gen_adj_common
- datasets/sensor_graph/describe_adjs.py

### 合成数据生成器

**generate_synth_data.py** — eclipse版本:
- 参考 walpurgis_cardgame/generate_synth_data.py 但做算法改动:
- 用 **Ornstein-Uhlenbeck过程** 生成时间相关噪声 (代替简单随机噪声)
- 邻接矩阵用 **k-NN based on 地理距离** (代替随机阈值)
- debug: 打印数据分布统计

### 配置文件 (configs/)

**SYNTH.yaml** — 要点:
```yaml
start_up:
  mode: scratch
  model_name: D2STGNN
  device: cpu
  load_pkl: False
data_args:
  data_dir: datasets/SYNTH
  adj_data_path: datasets/sensor_graph/adj_mx_synth.pkl
  adj_type: doubletransition
model_args:
  batch_size: 16
  num_feat: 1
  num_hidden: 16
  node_hidden: 8
  time_emb_dim: 8
  dropout: 0.1
  seq_length: 12
  k_t: 3
  k_s: 2
  gap: 3
  num_modalities: 2
optim_args:
  lrate: 0.002
  wdecay: 1.0e-5
  eps: 1.0e-8
  lr_schedule: True
  lr_sche_steps: [1, 5, 8]
  lr_decay_ratio: 0.5
  if_cl: True
  cl_epochs: 2
  output_seq_len: 12
  warm_epochs: 0
  epochs: 3
  patience: 100
  seq_length: 12
  print_model: False
```

也创建 METR-LA.yaml, PEMS-BAY.yaml, PEMS04.yaml, PEMS08.yaml (参考upstream configs但路径改为相对路径)

### __init__.py (顶层包)

```python
# walpurgis_eclipse — D2STGNN Eclipse variant
import os, sys, torch

_ECLIPSE_DEBUG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def _is_debug():
    return _ECLIPSE_DEBUG

def _dbg(tag, tensor_or_val, module="eclipse"):
    if not _ECLIPSE_DEBUG:
        return
    if isinstance(tensor_or_val, torch.Tensor):
        t = tensor_or_val
        msg = (f"[ECL:{tag}@{module}] shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f}")
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        if nan_count > 0:
            msg += f" *** NaN={nan_count} ***"
        if inf_count > 0:
            msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[ECL:{tag}@{module}] value={tensor_or_val}"
    print(msg, file=sys.stderr)


def snapshot_model(model, epoch=0, step=0):
    """全参数快照: grad_norm降序, nan检测"""
    if not _ECLIPSE_DEBUG:
        return
    print(f"\n[ECL] === Model Snapshot (epoch={epoch}, step={step}) ===", file=sys.stderr)
    params = []
    for name, p in model.named_parameters():
        info = {"name": name, "shape": list(p.shape), "mean": p.data.mean().item(),
                "std": p.data.std().item(), "has_nan": torch.isnan(p.data).any().item()}
        if p.grad is not None:
            info["grad_norm"] = p.grad.norm().item()
        params.append(info)
    params.sort(key=lambda x: x.get("grad_norm", 0), reverse=True)
    for p in params[:10]:
        print(f"  {p['name']}: shape={p['shape']} mean={p['mean']:.6f} std={p['std']:.6f} grad={p.get('grad_norm','N/A')}", file=sys.stderr)
    print(f"[ECL] === End Snapshot ===\n", file=sys.stderr)


def register_activation_hooks(model):
    """注册forward hooks追踪每层activation"""
    class _Tracker:
        def __init__(self):
            self.records = {}
            self.handles = []
        def _hook(self, name):
            def fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    self.records[name] = {
                        "mean": out.mean().item(), "std": out.std().item(),
                        "zero_frac": (out == 0).float().mean().item()
                    }
            return fn
        def report(self):
            print(f"\n[ECL] === Activation Report ({len(self.records)} layers) ===", file=sys.stderr)
            for name, r in self.records.items():
                flag = " *** DEAD ***" if r["zero_frac"] > 0.9 else ""
                print(f"  {name}: mean={r['mean']:.6f} std={r['std']:.6f} zero={r['zero_frac']:.2%}{flag}", file=sys.stderr)
            print(f"[ECL] === End Report ===\n", file=sys.stderr)
        def remove(self):
            for h in self.handles:
                h.remove()
    tracker = _Tracker()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(tracker._hook(name))
            tracker.handles.append(h)
    return tracker


def gradient_health_check(model):
    """检测梯度健康: 爆炸/消失/NaN"""
    if not _ECLIPSE_DEBUG:
        return
    print(f"\n[ECL] === Gradient Health Check ===", file=sys.stderr)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.norm().item()
        if gn > 100:
            print(f"  EXPLODING: {name} grad_norm={gn:.2f}", file=sys.stderr)
        elif gn < 1e-7:
            print(f"  VANISHING: {name} grad_norm={gn:.2e}", file=sys.stderr)
        if torch.isnan(p.grad).any():
            print(f"  NaN GRAD: {name}", file=sys.stderr)
    print(f"[ECL] === End Check ===\n", file=sys.stderr)
```

### main.py (入口)

参考 upstream/d2stgnn/main.py 但做以下改动:
- 支持 --debug flag 开启 ECLIPSE_DEBUG
- 支持 --epochs 覆盖config
- 支持 --device cpu/cuda
- DataParallel多GPU支持
- AMP混合精度
- 训练前activation probe
- 训练中每5epoch做gradient_health_check
- debug: 打印完整训练配置

## 第三步: 创建训练入口 (根目录)

**train_eclipse.py** — 参考 train_cardgame.py 但import改为 walpurgis_eclipse

**run_eclipse.sh** — 参考 run_cardgame.sh 但改为 ECLIPSE_DEBUG / train_eclipse.py

## 第四步: 验证

```bash
# 生成合成数据 + 训练2 epoch
EPOCHS=2 ECLIPSE_DEBUG=1 bash run_eclipse.sh
```

确保:
1. 合成数据生成成功
2. 模型可以forward pass
3. 训练循环可以跑完2个epoch
4. debug输出正常 (每个模块都有print)
5. 模型保存到 output/D2STGNN_SYNTH.pt

## 第五步: git push

```bash
git add -A
git commit --author="dylanyunlon <dogechat@163.com>" -m "feat(eclipse): D2STGNN Eclipse变体完整移植 — 算法改写+debug体系 [M700-M724]"
git push origin main
```

## 算法改写总结表 (必须全部实现)

| 模块 | upstream | eclipse | 改动类型 |
|------|----------|---------|----------|
| 损失函数 | masked_mae (L1) | Tukey biweight + gradient_penalty | 核心算法 |
| 估计门 | 2层FC+ReLU | 3层FC+Swish+ChannelSE+可学习τ | 架构重写 |
| 残差分解 | LayerNorm(x-ReLU(y)) | RMSNorm(x-Mish(y)*α), α可学习 | 激活+缩放 |
| 扩散卷积 | BN+ReLU | InstanceNorm2d+GELU+gconv残差skip | 归一化+激活 |
| 扩散块 | Linear backcast | 2层MLP+LayerNorm+sigmoid门控 | 门控机制 |
| 距离函数 | scaled-dot attention | cosine_sim+learnable_bias+LayerNorm | 注意力机制 |
| Mask | 硬mask | sigmoid soft-gating (nn.Parameter) | 软化 |
| 归一化器 | D⁻¹A行归一化 | D⁻½AD⁻½对称归一化 | 理论正确 |
| GRU层 | 裸GRU | GRU后接LayerNorm | 稳定性 |
| Transformer | 裸attention | 残差连接(out=X+attn) | 梯度流 |
| PE | 固定正弦 | 可学习phase offset | 灵活性 |
| Backcast | 直接减法 | sigmoid门控残差 | 自适应 |
| 层聚合 | sum() | EMA指数衰减聚合 | 长程依赖 |
| 输出头 | ReLU FC | GELU+SpectralNorm | 表达力 |
| Embedding | 直接Linear | Linear+Gaussian noise注入 | 正则化 |
| 优化器 | Adam+MultiStepLR | AdamW+ReduceLROnPlateau | 现代化 |
| 梯度裁剪 | 固定clip=5 | 自适应p95裁剪 | 鲁棒性 |
| CL | 线性增长 | 对数增长 | 课程平滑 |
| 合成数据 | 简单噪声 | Ornstein-Uhlenbeck过程 | 时序相关 |
| 数据shuffle | np.random.permutation | Fisher-Yates in-place | 效率 |

如果一条消息放不下所有文件，先创建核心模型文件并验证import，然后发"Continue"继续创建剩余文件。

## 接力信息

| Claude # | 里程碑 | 内容 | 状态 |
|----------|--------|------|------|
| 第一位Claude(当前对话) | M694-M699 | 规划+派发eclipse任务 | ✅ 进行中 |
| **第四位Claude(你)** | **M700-M724** | **walpurgis_eclipse完整移植** | **⏳ 你的任务** |
| 第五位Claude | M725-M749 | walpurgis_aurora变体 | ⏳ 待分配 |
| 第六位Claude | M750-M774 | 全变体对比实验 | ⏳ 待分配 |
