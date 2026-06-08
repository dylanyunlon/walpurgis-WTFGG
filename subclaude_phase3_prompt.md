# 子Claude任务调度Prompt — Phase 3 SOTA实验
# 用于 claude_hk_chat.sh 发送给 Opus 4.6 (medium)
# 发送时选择模型: claude oppus 4.6 (medium)
# 截断时发送: Continue

## 你的角色
你是Walpurgis-WTFGG项目的实验运行Claude。你的任务是在GPU服务器上运行D2STGNN变体的实验，让结果超越SOTA (STAEFormer MAE=2.90 on METR-LA)。

## 仓库信息
- GitHub: https://github.com/dylanyunlon/walpurgis-WTFGG
- Token: <GH_TOKEN>
- Git author: dylanyunlon <dogechat@163.com>
- 分支: main (唯一分支, 不要开新分支)
- 不要加v2/v3/port等后缀到任何文件或目录名

## 第一步: 检查GPU环境
```bash
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
free -h
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
nvcc --version 2>/dev/null | tail -1
```

## 第二步: Clone仓库并配置环境
```bash
export GH_TOKEN="<GH_TOKEN>"
git clone https://${GH_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
```

## 第三步: Conda环境
如果已有walpurgis环境则复用:
```bash
conda activate walpurgis
```
否则创建:
```bash
conda create -n walpurgis python=3.10 -y
conda activate walpurgis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pyyaml scikit-learn
```

## 第四步: 下载数据集
METR-LA和PEMS-BAY数据集需要从upstream下载:
```bash
# METR-LA (207 nodes)
cd datasets
mkdir -p METR-LA
# 从 https://zenodo.org/records/5146275 下载 metr-la.h5
# 或从 upstream D2STGNN 的 Google Drive 链接下载
python ../upstream/d2stgnn/datasets/raw_data/METR-LA/generate_training_data.py

# PEMS-BAY (325 nodes)  
mkdir -p PEMS-BAY
python ../upstream/d2stgnn/datasets/raw_data/PEMS-BAY/generate_training_data.py

# sensor graph adjacency
mkdir -p sensor_graph
# adj_mx_la.pkl 和 adj_mx_bay.pkl 需要从upstream获取
cd ..
```

## 第五步: 运行实验
高潜力变体排名: cascade > nebula > prism > flux > reverie

你的具体任务 (根据你的M编号):

### M110-M115 (第二位Claude):
```bash
bash gpu_experiment.sh cascade METR-LA cuda:0 80
# 结果会自动push
```

### M116-M121 (第三位Claude):
```bash
bash gpu_experiment.sh nebula METR-LA cuda:0 80
bash gpu_experiment.sh prism METR-LA cuda:0 80
bash gpu_experiment.sh nebula PEMS-BAY cuda:0 80
bash gpu_experiment.sh prism PEMS-BAY cuda:0 80
```

### M122-M127 (第四位Claude):
```bash
bash gpu_experiment.sh flux METR-LA cuda:0 80
bash gpu_experiment.sh reverie METR-LA cuda:0 80
# upstream baseline验证
cd upstream/d2stgnn && python main.py --dataset METR-LA
bash gpu_experiment.sh flux PEMS-BAY cuda:0 80
```

### M128-M133 (第五位Claude):
```bash
git pull origin main  # 获取前面Claude的实验结果
# 读取所有 output/results_*/summary.json
# 生成 output/comparison_table.tex
# 确定最优变体, 微调超参数
```

### M134-M139 (第六位Claude):
```bash
git pull origin main
# 验证实验可复现
# 更新论文tex Section 5
# 仓库最终清理
```

## SOTA目标参考
METR-LA数据集当前SOTA:
| Model       | MAE  | RMSE | MAPE  |
|-------------|------|------|-------|
| STAEFormer  | 2.90 | 5.91 | 8.12% |
| PDFormer    | 2.94 | 6.08 | 8.56% |
| STG-NCDE    | 2.96 | 6.51 | 9.13% |
| D2STGNN     | 3.04 | 6.23 | 8.33% |

我们的目标: MAE < 2.85

## 严格规则:
1. 改算法, 不改字符串/docstring
2. 直接push到main, 不开新分支
3. 不加v2/v3/port/backup任何后缀
4. git author: dylanyunlon <dogechat@163.com>
5. 每个实验必须跑完整epoch (不要提前中断)
6. 结果自动push (gpu_experiment.sh已内置)
7. 如果被截断, 对方会发送Continue, 你继续执行

## 当前仓库结构 (关键部分):
```
walpurgis-WTFGG/
├── src/
│   ├── walpurgis_cascade/   # 潜力: very_high
│   ├── walpurgis_nebula/    # 潜力: very_high
│   ├── walpurgis_prism/     # 潜力: very_high
│   ├── walpurgis_flux/      # 潜力: high
│   ├── walpurgis_reverie/   # 潜力: high (最新)
│   └── ... (20+ other variants)
├── upstream/d2stgnn/         # 原始D2STGNN
├── bench/sota.json           # SOTA基准数据
├── gpu_experiment.sh         # GPU实验自动runner
├── train_*.py                # 每个变体的训练入口
├── run_*.sh                  # 每个变体的运行脚本
└── MULTI_CLAUDE_PLAN.md      # 完整开发计划
```

现在开始执行你的任务。先检查GPU环境, 然后按步骤进行。
