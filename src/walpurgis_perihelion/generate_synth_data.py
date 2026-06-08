"""generate_synth_data.py — Perihelion合成数据生成"""
import os
import sys
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_synth_traffic(num_nodes=10, num_timesteps=500,
                           num_feat=1, seq_x=12, seq_y=12,
                           output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[PERI-SYNTH] Generating: {num_nodes} nodes, {num_timesteps} steps")
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    np.random.seed(42)
    node_offsets = np.random.randn(num_nodes) * 5
    node_scales = 1 + np.random.randn(num_nodes) * 0.1
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        base = (daily + weekly) * node_scales[n] + node_offsets[n] + 60
        noise = np.random.randn(num_timesteps) * 3
        data[:, n, 0] = base + noise
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))
    x_offsets = np.arange(-(seq_x - 1), 1, 1)
    y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for t_idx in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[t_idx + x_offsets, ...])
        ys.append(data[t_idx + y_offsets, ..., :1])
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
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
        np.savez_compressed(os.path.join(output_dir, f"{name}.npz"), x=_x, y=_y,
                            x_offsets=x_offsets.reshape(-1, 1), y_offsets=y_offsets.reshape(-1, 1))
        print(f"  {name}: x={_x.shape} y={_y.shape}")
    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    adj[adj < 0.7] = 0
    np.fill_diagonal(adj, 0)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj>0].size / adj.size:.2%}")
    mean = splits['train'][0][..., 0].mean()
    std = splits['train'][0][..., 0].std()
    pickle.dump({'mean': mean, 'std': std}, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")
    import json
    checksums = {}
    for name, (_x, _y) in splits.items():
        checksums[name] = {'x_mean': float(_x.mean()), 'x_std': float(_x.std()),
                           'y_mean': float(_y.mean()), 'y_std': float(_y.std()),
                           'x_shape': list(_x.shape), 'y_shape': list(_y.shape)}
    with open(os.path.join(output_dir, 'checksums.json'), 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"[PERI-SYNTH] Done. Output: {output_dir}")
    return output_dir

if __name__ == '__main__':
    generate_synth_traffic()
