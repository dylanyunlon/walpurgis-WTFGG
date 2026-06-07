"""Eclipse synth data: Ornstein-Uhlenbeck process + k-NN adjacency."""
import os, sys, hashlib, json, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def _ornstein_uhlenbeck(T, N, theta=0.7, mu=0.0, sigma=3.0, seed=42):
    """Generate temporally correlated noise via OU process.
    dX = theta*(mu - X)*dt + sigma*dW"""
    rng = np.random.RandomState(seed)
    dt = 1.0; X = np.zeros((T, N))
    X[0] = rng.randn(N) * sigma
    for t in range(1, T):
        dW = rng.randn(N) * np.sqrt(dt)
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    return X

def _knn_adjacency(num_nodes, k=3, seed=42):
    """Build adjacency via k-NN on random spatial coordinates."""
    rng = np.random.RandomState(seed)
    coords = rng.rand(num_nodes, 2) * 100  # random 2D positions
    from scipy.spatial.distance import cdist
    dist = cdist(coords, coords)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        neighbors = np.argsort(dist[i])[1:k+1]
        for j in neighbors:
            w = np.exp(-dist[i, j] / 20.0)
            adj[i, j] = w; adj[j, i] = w
    np.fill_diagonal(adj, 0)
    return adj

def generate_synth_traffic(num_nodes=10, num_timesteps=500, num_feat=1, seq_x=12, seq_y=12, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ECL-SYNTH] Generating: {num_nodes} nodes, {num_timesteps} steps")
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    np.random.seed(42)
    offsets = np.random.randn(num_nodes) * 5
    scales = 1 + np.random.randn(num_nodes) * 0.1
    # Ornstein-Uhlenbeck noise (vs simple random noise)
    ou_noise = _ornstein_uhlenbeck(num_timesteps, num_nodes, theta=0.7, sigma=3.0, seed=42)
    if _ECL_DBG: print(f"[ECL:synth] OU noise: mean={ou_noise.mean():.4f} std={ou_noise.std():.4f}", file=sys.stderr)
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        data[:, n, 0] = (daily + weekly) * scales[n] + offsets[n] + 60 + ou_noise[:, n]
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))
    x_offsets = np.arange(-(seq_x - 1), 1, 1); y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for ti in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[ti + x_offsets, ...]); ys.append(data[ti + y_offsets, ..., :1])
    x = np.stack(xs); y = np.stack(ys)
    n_total = x.shape[0]; n_train = int(n_total * 0.7); n_test = int(n_total * 0.2); n_val = n_total - n_train - n_test
    splits = {'train': (x[:n_train], y[:n_train]), 'val': (x[n_train:n_train+n_val], y[n_train:n_train+n_val]), 'test': (x[-n_test:], y[-n_test:])}
    checksums = {}
    for name, (_x, _y) in splits.items():
        p = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(p, x=_x, y=_y)
        print(f"  {name}: x={_x.shape} y={_y.shape}")
        with open(p, 'rb') as fh: checksums[name] = hashlib.sha256(fh.read()).hexdigest()
    json.dump(checksums, open(os.path.join(output_dir, 'checksums.json'), 'w'), indent=2)
    # k-NN adjacency (vs random threshold)
    adj = _knn_adjacency(num_nodes, k=3, seed=42)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj>0].size/adj.size:.2%}")
    mean = splits['train'][0][..., 0].mean(); std = splits['train'][0][..., 0].std()
    pickle.dump({'mean': mean, 'std': std}, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")
    print(f"[ECL-SYNTH] Done: {output_dir}")
    return output_dir

if __name__ == '__main__': generate_synth_traffic()
