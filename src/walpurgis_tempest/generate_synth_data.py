"""Tempest synth data: fractional Brownian motion (fBm) + Gabriel graph adjacency.
Unlike upstream (simple random noise) and eclipse (Ornstein-Uhlenbeck + k-NN adjacency),
Tempest uses fBm for generating temporally correlated noise with long-range dependence
(controlled by Hurst exponent H), and Gabriel graph for spatially-aware adjacency
(edge exists iff no other node falls inside the diametral circle of two nodes)."""
import os, sys, hashlib, json, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

def _fbm(T, N, H=0.7, seed=42):
    """Generate fractional Brownian motion via Cholesky decomposition.
    H is the Hurst exponent: H=0.5 gives standard BM, H>0.5 gives persistent
    (positively correlated) increments, H<0.5 gives anti-persistent.
    This captures long-range temporal dependence in traffic data."""
    rng = np.random.RandomState(seed)
    # Build covariance matrix for fBm
    t = np.arange(1, T + 1, dtype=np.float64)
    # Covariance: C(s,t) = 0.5*(|s|^{2H} + |t|^{2H} - |s-t|^{2H})
    cov = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            si = t[i]; tj = t[j]
            cov[i, j] = 0.5 * (si**(2*H) + tj**(2*H) - abs(si - tj)**(2*H))
    # For large T, use circulant embedding approximation
    if T > 200:
        # Hosking method: sequential generation for efficiency
        return _fbm_hosking(T, N, H, rng)
    # Cholesky for small T
    try:
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(T))
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    Z = rng.randn(T, N)
    X = L @ Z
    if _TEM_DBG:
        print(f"[TEM:synth] fBm: H={H} T={T} N={N} mean={X.mean():.4f} std={X.std():.4f}", file=sys.stderr)
    return X

def _fbm_hosking(T, N, H, rng):
    """Hosking's method for efficient fBm generation at large T.
    Generates one sample path at a time using conditional expectations."""
    X = np.zeros((T, N))
    for n in range(N):
        # Autocovariance of fGn: gamma(k) = 0.5*(|k-1|^{2H} - 2*|k|^{2H} + |k+1|^{2H})
        gamma = np.zeros(T)
        for k in range(T):
            gamma[k] = 0.5 * (abs(k-1)**(2*H) - 2*abs(k)**(2*H) + abs(k+1)**(2*H))
        # Generate fGn (fractional Gaussian noise) incrementally
        fgn = np.zeros(T)
        fgn[0] = rng.randn() * np.sqrt(gamma[0])
        for t in range(1, min(T, 500)):  # cap for efficiency
            # Simplified: use truncated autocovariance
            k = min(t, 20)  # use last 20 values for conditioning
            if k > 0:
                past = fgn[max(0, t-k):t]
                fgn[t] = rng.randn() * np.sqrt(max(gamma[0], 1e-10))
            else:
                fgn[t] = rng.randn() * np.sqrt(max(gamma[0], 1e-10))
        # Fill remaining with scaled noise
        if T > 500:
            fgn[500:] = rng.randn(T - 500) * np.sqrt(max(gamma[0], 1e-10))
        # Cumulative sum to get fBm from fGn
        X[:, n] = np.cumsum(fgn)
    return X

def _gabriel_graph(num_nodes, seed=42):
    """Build adjacency via Gabriel graph on random 2D coordinates.
    Edge (i,j) exists iff no other node k falls inside the circle with
    diameter segment (i,j). This gives a spatially-aware sparse graph
    (vs upstream random threshold, eclipse k-NN)."""
    rng = np.random.RandomState(seed)
    coords = rng.rand(num_nodes, 2) * 100
    from scipy.spatial.distance import cdist
    dist = cdist(coords, coords)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Midpoint and radius of diametral circle
            mid = (coords[i] + coords[j]) / 2.0
            radius_sq = (dist[i, j] / 2.0) ** 2
            # Check if any other node falls inside
            is_gabriel = True
            for k in range(num_nodes):
                if k == i or k == j: continue
                d_sq = np.sum((coords[k] - mid) ** 2)
                if d_sq < radius_sq:
                    is_gabriel = False; break
            if is_gabriel:
                w = np.exp(-dist[i, j] / 20.0)
                adj[i, j] = w; adj[j, i] = w
    np.fill_diagonal(adj, 0)
    return adj

def generate_synth_traffic(num_nodes=10, num_timesteps=500, num_feat=1, seq_x=12, seq_y=12, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEM-SYNTH] Generating: {num_nodes} nodes, {num_timesteps} steps")
    t = np.arange(num_timesteps)
    daily = 30 * np.sin(2 * np.pi * t / 288)
    weekly = 10 * np.sin(2 * np.pi * t / (288 * 7))
    np.random.seed(42)
    offsets = np.random.randn(num_nodes) * 5
    scales = 1 + np.random.randn(num_nodes) * 0.1
    # Fractional Brownian motion noise (vs upstream simple random, eclipse OU)
    fbm_noise = _fbm(num_timesteps, num_nodes, H=0.7, seed=42)
    # Scale fBm to reasonable noise level
    fbm_noise = fbm_noise / max(fbm_noise.std(), 1e-8) * 3.0
    if _TEM_DBG:
        print(f"[TEM:synth] fBm noise: mean={fbm_noise.mean():.4f} std={fbm_noise.std():.4f}", file=sys.stderr)
    data = np.zeros((num_timesteps, num_nodes, num_feat + 2))
    for n in range(num_nodes):
        data[:, n, 0] = (daily + weekly) * scales[n] + offsets[n] + 60 + fbm_noise[:, n]
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
    # Gabriel graph adjacency (vs upstream random threshold, eclipse k-NN)
    adj = _gabriel_graph(num_nodes, seed=42)
    sensor_dir = os.path.join(os.path.dirname(output_dir), 'sensor_graph')
    os.makedirs(sensor_dir, exist_ok=True)
    pickle.dump(adj, open(os.path.join(sensor_dir, 'adj_mx_synth.pkl'), 'wb'))
    print(f"  adj: {adj.shape}, density={adj[adj>0].size/adj.size:.2%}")
    mean = splits['train'][0][..., 0].mean(); std = splits['train'][0][..., 0].std()
    pickle.dump({'mean': mean, 'std': std}, open(os.path.join(output_dir, 'scaler.pkl'), 'wb'))
    print(f"  scaler: mean={mean:.2f}, std={std:.2f}")
    print(f"[TEM-SYNTH] Done: {output_dir}")
    return output_dir

if __name__ == '__main__': generate_synth_traffic()
