"""
generate_synth_data.py — CardGame合成数据生成
生成小规模假数据用于smoke test和调试

算法改写 (vs nightfall):
  1. 空间相关矩阵: 节点间流量通过Cholesky分解产生空间相关性
  2. 异常注入: 随机在5%的时间步注入spike异常
  3. train/val/test SHA-256完整性校验

运行: cd walpurgis-WTFGG && PYTHONPATH=src python -m walpurgis_cardgame.generate_synth_data
"""
import os
import sys
import hashlib
import json
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _dbg(tag, tensor, module="generate_synth"):
    if not _CG_DEBUG:
        return
    if hasattr(tensor, 'shape'):
        arr = np.asarray(tensor)
        msg = (f"[CG-DBG:{tag}] shape={list(arr.shape)} dtype={arr.dtype} "
               f"min={arr.min():.6f} max={arr.max():.6f} "
               f"mean={arr.mean():.6f} std={arr.std():.6f}")
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            msg += f" *** NaN={nan_count} ***"
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def _make_spatial_corr(num_nodes, seed=42):
    """生成空间相关矩阵 (Cholesky分解)"""
    rng = np.random.RandomState(seed)
    A = rng.randn(num_nodes, num_nodes) * 0.3
    cov = A @ A.T + np.eye(num_nodes) * 1.0
    # 归一化为相关矩阵
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    L = np.linalg.cholesky(corr)
    return L


def _inject_anomalies(data, anomaly_ratio=0.05, spike_factor=3.0, seed=123):
    """在随机时间步注入spike异常"""
    rng = np.random.RandomState(seed)
    T = data.shape[0]
    n_anomaly = int(T * anomaly_ratio)
    anomaly_idx = rng.choice(T, size=n_anomaly, replace=False)
    for idx in anomaly_idx:
        node = rng.randint(0, data.shape[1])
        data[idx, node, 0] *= spike_factor
    return data, anomaly_idx


def generate_synth_traffic(num_nodes=10, num_timesteps=500, num_feat=1,
                           seq_x=12, seq_y=12, output_dir=None):
    """
    生成合成交通数据,格式与METR-LA/PEMS-BAY一致
    CardGame特色: 空间相关节点 + 异常注入
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[CG-SYNTH] Generating synthetic data: {num_nodes} nodes, {num_timesteps} steps")

    # 生成带周期性的交通流量数据
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))

    np.random.seed(42)
    node_offsets = np.random.randn(num_nodes) * 5
    node_scales = 1 + np.random.randn(num_nodes) * 0.1

    # 空间相关噪声 (Cholesky)
    L = _make_spatial_corr(num_nodes, seed=42)
    white_noise = np.random.randn(num_timesteps, num_nodes)
    correlated_noise = (white_noise @ L.T) * 3.0

    if _CG_DEBUG:
        _dbg("spatial_corr_L", L)
        _dbg("correlated_noise", correlated_noise)

    # 构造数据: [T, N, F]
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        base = (daily + weekly) * node_scales[n] + node_offsets[n] + 60
        data[:, n, 0] = base + correlated_noise[:, n]

    # 异常注入
    data, anomaly_idx = _inject_anomalies(data, anomaly_ratio=0.05)
    print(f"  Injected {len(anomaly_idx)} anomaly timesteps")
    if _CG_DEBUG:
        _dbg("data_with_anomalies", data[:, :, 0])

    # time_in_day (0~1)
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    # day_in_week (0~6)
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))

    _dbg("synth_data_full", data)

    # 构造滑窗
    x_offsets = np.arange(-(seq_x - 1), 1, 1)
    y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for t_idx in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[t_idx + x_offsets, ...])
        ys.append(data[t_idx + y_offsets, ..., :1])
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)

    # 70/10/20 split
    n_total = x.shape[0]
    n_train = int(n_total * 0.7)
    n_test = int(n_total * 0.2)
    n_val = n_total - n_train - n_test

    splits = {
        'train': (x[:n_train], y[:n_train]),
        'val': (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test': (x[-n_test:], y[-n_test:]),
    }

    checksums = {}
    for name, (_x, _y) in splits.items():
        out_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(out_path,
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(-1, 1),
                            y_offsets=y_offsets.reshape(-1, 1))
        print(f"  {name}: x={_x.shape} y={_y.shape}")
        # SHA-256 checksum
        with open(out_path, 'rb') as fh:
            checksums[name] = hashlib.sha256(fh.read()).hexdigest()

    # 保存checksum
    cksum_path = os.path.join(output_dir, "checksums.json")
    with open(cksum_path, 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"  checksums saved: {cksum_path}")

    # 生成邻接矩阵
    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    adj[adj < 0.7] = 0
    np.fill_diagonal(adj, 0)

    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj > 0].size / adj.size:.2%}")
    _dbg("synth_adj", adj)

    # 保存scaler
    mean = splits['train'][0][..., 0].mean()
    std = splits['train'][0][..., 0].std()
    scaler = {'mean': mean, 'std': std}
    pickle.dump(scaler, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")

    print(f"[CG-SYNTH] Done. Output: {output_dir}")
    return output_dir


if __name__ == '__main__':
    generate_synth_traffic()
