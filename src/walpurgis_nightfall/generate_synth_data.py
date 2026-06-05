"""
generate_synth_data.py — Nightfall合成数据生成
生成小规模假数据用于smoke test和调试

运行: cd walpurgis-WTFGG && PYTHONPATH=src python -m walpurgis_nightfall.generate_synth_data
"""
import os
import sys
import numpy as np
import pickle

# 确保可从项目根运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def generate_synth_traffic(num_nodes=10, num_timesteps=500, num_feat=1,
                           seq_x=12, seq_y=12, output_dir=None):
    """
    生成合成交通数据,格式与METR-LA/PEMS-BAY一致
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[NF-SYNTH] Generating synthetic data: {num_nodes} nodes, {num_timesteps} steps")
    # 生成带周期性的交通流量数据
    t = np.arange(num_timesteps)
    # 日周期 (288个时间步/天)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    # 周周期
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    # 节点间有微小差异
    np.random.seed(42)
    node_offsets = np.random.randn(num_nodes) * 5
    node_scales = 1 + np.random.randn(num_nodes) * 0.1
    # 构造数据: [T, N, F]
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))  # +time_in_day +day_in_week
    for n in range(num_nodes):
        base = (daily + weekly) * node_scales[n] + node_offsets[n] + 60
        noise = np.random.randn(num_timesteps) * 3
        data[:, n, 0] = base + noise
    # time_in_day (0~1)
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    # day_in_week (0~6)
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))
    # 构造滑窗
    x_offsets = np.arange(-(seq_x - 1), 1, 1)
    y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for t_idx in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[t_idx + x_offsets, ...])
        ys.append(data[t_idx + y_offsets, ..., :1])  # y只取第一个feat
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    # 70/10/20 split
    n = x.shape[0]
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    splits = {
        'train': (x[:n_train], y[:n_train]),
        'val': (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test': (x[-n_test:], y[-n_test:]),
    }
    for name, (_x, _y) in splits.items():
        np.savez_compressed(os.path.join(output_dir, f"{name}.npz"),
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(-1, 1),
                            y_offsets=y_offsets.reshape(-1, 1))
        print(f"  {name}: x={_x.shape} y={_y.shape}")
    # 生成邻接矩阵
    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    adj[adj < 0.7] = 0
    np.fill_diagonal(adj, 0)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj>0].size / adj.size:.2%}")
    # 保存scaler
    mean = splits['train'][0][..., 0].mean()
    std = splits['train'][0][..., 0].std()
    scaler = {'mean': mean, 'std': std}
    pickle.dump(scaler, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")
    print(f"[NF-SYNTH] Done. Output: {output_dir}")
    return output_dir


if __name__ == '__main__':
    generate_synth_traffic()
