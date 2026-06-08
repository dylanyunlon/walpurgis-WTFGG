# 你是第二位Claude (Opus 4.6 medium)，负责在GPU服务器上运行实验

## 第一步: 同步配置
```bash
git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config
```

## 第二步: 克隆仓库
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
```

## 第三步: 查看服务器资源
```bash
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
free -h
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
nvidia-smi topo -m 2>/dev/null || echo "topo not available"
uname -r
cat /etc/os-release 2>/dev/null | head -3
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"
```

## 第四步: 配置conda环境
```bash
# 如果已有walpurgis环境则复用
conda activate walpurgis 2>/dev/null || {
    conda create -n walpurgis python=3.10 -y
    conda activate walpurgis
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy scipy pyyaml setproctitle matplotlib pandas
}
```

## 第五步: 下载METR-LA数据集
METR-LA是交通流量数据集, 207个传感器节点, 34272个时间步。
标准预处理格式 (来自DCRNN):
- datasets/METR-LA/{train,val,test}.npz
- datasets/sensor_graph/adj_mx_la.pkl

下载方法:
```bash
# 方法1: LibCity项目
pip install gdown
python3 -c "
import gdown, os
# METR-LA processed data (Google Drive ID from LibCity)
url = 'https://drive.google.com/uc?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC'
gdown.download(url, 'metr-la.zip', quiet=False)
import zipfile
with zipfile.ZipFile('metr-la.zip') as z:
    z.extractall('datasets/')
"

# 方法2: 从DCRNN原始数据预处理
git clone https://github.com/liyaguang/DCRNN.git /tmp/dcrnn
cd /tmp/dcrnn
python generate_training_data.py --output_dir /path/to/datasets/METR-LA --traffic_df_filename data/metr-la.h5
```

## 第六步: 运行实验
```bash
cd ~/walpurgis-WTFGG
# 确认数据存在
ls -la datasets/METR-LA/
ls -la datasets/sensor_graph/adj_mx_la.pkl

# 运行cathexis在METR-LA
./run_experiment.sh cathexis METR-LA cuda:0 80
```

## 第七步: Push结果
```bash
TOKEN="${GITHUB_TOKEN}"
git remote set-url origin "https://${TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git"
git add -A
git commit -m "result(cathexis): METR-LA experiment results"
git push origin main
```

## 关键规则
1. 改的是算法，不是字符串/docstring — 不要做str_replace改名字
2. 直接push到main — 不开新分支，不加v2/v3/port后缀
3. 如果被截断，对方会发送Continue继续
4. 结果JSON必须包含: variant, dataset, avg_MAE, avg_RMSE, avg_MAPE, per_horizon
5. 目标: 超越STAEFormer MAE=2.90 on METR-LA

## 对比目标 (METR-LA)
| Model | Year | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| STAEFormer | 2023 | 2.90 | 5.91 | 8.12% |
| D2STGNN | 2022 | 3.04 | 6.23 | 8.33% |
