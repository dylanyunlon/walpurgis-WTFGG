"""Nebula synth data: Lévy flight process + importance sampling shuffle."""
import os, sys, hashlib, json, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


def _levy_flight(T, N, alpha=1.5, scale=2.0, seed=42):
    """Generate Lévy flight noise via stable distribution approximation.
    Uses Chambers-Mallows-Stuck method to generate alpha-stable random variables.
    Lévy flights produce heavy-tailed jumps, modeling sudden traffic spikes."""
    rng = np.random.RandomState(seed)
    # CMS method for alpha-stable
    U = rng.uniform(-np.pi/2, np.pi/2, size=(T, N))
    W = rng.exponential(1.0, size=(T, N))
    # Symmetric alpha-stable
    if alpha == 2.0:
        X = rng.randn(T, N) * scale
    else:
        num = np.sin(alpha * U)
        den = np.cos(U) ** (1.0 / alpha)
        exp_term = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
        X = scale * num / den * exp_term
    # Clip extreme values for numerical stability
    X = np.clip(X, -50 * scale, 50 * scale)
    # Cumulative sum for flight trajectory
    X = np.cumsum(X, axis=0) * 0.05  # scale down for reasonable traffic values
    if _NEB_DBG: print(f"[NEB:synth] Levy flight: alpha={alpha} mean={X.mean():.4f} std={X.std():.4f}", file=sys.stderr)
    return X


def _importance_sampling_split(x, y, train_frac=0.7, test_frac=0.2, seed=42):
    """Importance sampling for data split: weight samples by their variance
    to ensure high-variance (informative) samples are well-represented in train set."""
    rng = np.random.RandomState(seed)
    n_total = x.shape[0]
    n_train = int(n_total * train_frac)
    n_test = int(n_total * test_frac)
    n_val = n_total - n_train - n_test
    # Compute per-sample variance as importance weight
    sample_var = np.var(x[:, :, :, 0], axis=(1, 2))  # [n_total]
    # Softmax importance weights
    weights = np.exp(sample_var - sample_var.max())
    weights /= weights.sum()
    # Importance-weighted sampling for train set (without replacement)
    train_idx = rng.choice(n_total, size=n_train, replace=False, p=weights)
    remaining = np.setdiff1d(np.arange(n_total), train_idx)
    # Remaining split sequentially for val/test
    val_idx = remaining[:n_val]
    test_idx = remaining[n_val:n_val + n_test]
    if _NEB_DBG:
        print(f"[NEB:synth] Importance split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", file=sys.stderr)
        print(f"[NEB:synth] Train variance: mean={sample_var[train_idx].mean():.4f} vs overall={sample_var.mean():.4f}", file=sys.stderr)
    return {
        'train': (x[train_idx], y[train_idx]),
        'val': (x[val_idx], y[val_idx]),
        'test': (x[test_idx], y[test_idx])
    }


def _knn_adjacency(num_nodes, k=3, seed=42):
    """Build adjacency via k-NN on random spatial coordinates."""
    rng = np.random.RandomState(seed)
    coords = rng.rand(num_nodes, 2) * 100
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
    print(f"[NEB-SYNTH] Generating: {num_nodes} nodes, {num_timesteps} steps")
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    np.random.seed(42)
    offsets = np.random.randn(num_nodes) * 5
    scales = 1 + np.random.randn(num_nodes) * 0.1
    # Nebula: Lévy flight noise (heavy-tailed jumps)
    levy_noise = _levy_flight(num_timesteps, num_nodes, alpha=1.5, scale=2.0, seed=42)
    if _NEB_DBG: print(f"[NEB:synth] Levy noise: mean={levy_noise.mean():.4f} std={levy_noise.std():.4f}", file=sys.stderr)
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        data[:, n, 0] = (daily + weekly) * scales[n] + offsets[n] + 60 + levy_noise[:, n]
    data[:, :, 1] = np.tile((t % 288 / 288).reshape(-1, 1), (1, num_nodes))
    data[:, :, 2] = np.tile((t // 288 % 7).reshape(-1, 1), (1, num_nodes))
    x_offsets = np.arange(-(seq_x - 1), 1, 1); y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for ti in range(seq_x - 1, num_timesteps - seq_y):
        xs.append(data[ti + x_offsets, ...]); ys.append(data[ti + y_offsets, ..., :1])
    x = np.stack(xs); y = np.stack(ys)
    # Nebula: importance sampling split
    splits = _importance_sampling_split(x, y, train_frac=0.7, test_frac=0.2, seed=42)
    checksums = {}
    for name, (_x, _y) in splits.items():
        p = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(p, x=_x, y=_y)
        print(f"  {name}: x={_x.shape} y={_y.shape}")
        with open(p, 'rb') as fh: checksums[name] = hashlib.sha256(fh.read()).hexdigest()
    json.dump(checksums, open(os.path.join(output_dir, 'checksums.json'), 'w'), indent=2)
    adj = _knn_adjacency(num_nodes, k=3, seed=42)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj>0].size/adj.size:.2%}")
    mean = splits['train'][0][..., 0].mean(); std = splits['train'][0][..., 0].std()
    pickle.dump({'mean': mean, 'std': std}, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")
    print(f"[NEB-SYNTH] Done: {output_dir}")
    return output_dir


if __name__ == '__main__': generate_synth_traffic()
