import os, sys, numpy as np, pickle

def generate_synth_traffic(num_nodes=10, num_timesteps=500, num_feat=1,
                           seq_x=12, seq_y=12, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[CX-SYNTH] {num_nodes} nodes, {num_timesteps} steps")
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    np.random.seed(42)
    node_offsets = np.random.randn(num_nodes) * 5
    node_scales = 1 + np.random.randn(num_nodes) * 0.1
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        data[:, n, 0] = (daily + weekly) * node_scales[n] + node_offsets[n] + 60 + np.random.randn(num_timesteps) * 3
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))
    x_offsets = np.arange(-(seq_x - 1), 1, 1)
    y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for t_idx in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[t_idx + x_offsets, ...]); ys.append(data[t_idx + y_offsets, ..., :1])
    x, y = np.stack(xs), np.stack(ys)
    n_train, n_test = int(x.shape[0]*0.7), int(x.shape[0]*0.2)
    splits = {'train': (x[:n_train], y[:n_train]),
              'val': (x[n_train:n_train+(x.shape[0]-n_train-n_test)], y[n_train:n_train+(x.shape[0]-n_train-n_test)]),
              'test': (x[-n_test:], y[-n_test:])}
    for name, (_x, _y) in splits.items():
        np.savez_compressed(os.path.join(output_dir, f"{name}.npz"), x=_x, y=_y,
                            x_offsets=x_offsets.reshape(-1,1), y_offsets=y_offsets.reshape(-1,1))
        print(f"  {name}: x={_x.shape} y={_y.shape}")
    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2; adj[adj < 0.7] = 0; np.fill_diagonal(adj, 0)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    scaler = {'mean': splits['train'][0][...,0].mean(), 'std': splits['train'][0][...,0].std()}
    pickle.dump(scaler, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"[CX-SYNTH] Done.")

if __name__ == '__main__': generate_synth_traffic()
