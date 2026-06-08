"""Meridian — Generate synthetic (SYNTH) dataset for smoke testing.
Creates a small graph with fractional Brownian motion traffic signals.
"""
import numpy as np
import pickle
import os
import sys

def fbm_1d(n, hurst=0.7):
    """Approximate fBm via Cholesky of covariance matrix."""
    t = np.arange(1, n + 1, dtype=np.float64)
    cov = 0.5 * (np.abs(t[:, None]) ** (2*hurst) +
                  np.abs(t[None, :]) ** (2*hurst) -
                  np.abs(t[:, None] - t[None, :]) ** (2*hurst))
    cov += np.eye(n) * 1e-8
    L = np.linalg.cholesky(cov)
    z = np.random.randn(n)
    return L @ z

def generate_synth(num_nodes=20, num_steps=2016, hurst=0.7, seed=42):
    np.random.seed(seed)
    # adjacency
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj[adj < 0.6] = 0
    np.fill_diagonal(adj, 0)
    adj = (adj > 0).astype(np.float32)
    # traffic signals via fBm
    data = np.zeros((num_steps, num_nodes, 1), dtype=np.float32)
    for i in range(num_nodes):
        signal = fbm_1d(num_steps, hurst)
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        signal = signal * 20 + 60
        data[:, i, 0] = signal.astype(np.float32)
    return data, adj

def make_splits(data, adj, out_dir, window=12, horizon=12):
    os.makedirs(out_dir, exist_ok=True)
    num_samples = data.shape[0]
    # add time features
    tod = (np.arange(num_samples) % 288) / 288.0
    tod = np.tile(tod, [1, data.shape[1], 1]).transpose((2, 1, 0))
    dow = np.zeros((num_samples, data.shape[1], 7), dtype=np.float32)
    dow[np.arange(num_samples), :, (np.arange(num_samples) // 288) % 7] = 1
    full = np.concatenate([data, tod, dow], axis=-1)
    x_off = np.arange(-(window - 1), 1)
    y_off = np.arange(1, horizon + 1)
    xs, ys = [], []
    for t in range(abs(min(x_off)), num_samples - max(y_off)):
        xs.append(full[t + x_off])
        ys.append(full[t + y_off])
    xs, ys = np.stack(xs), np.stack(ys)
    n = xs.shape[0]
    nt = round(n * 0.2)
    nr = round(n * 0.7)
    nv = n - nt - nr
    np.savez_compressed(os.path.join(out_dir, 'train.npz'), x=xs[:nr], y=ys[:nr])
    np.savez_compressed(os.path.join(out_dir, 'val.npz'), x=xs[nr:nr+nv], y=ys[nr:nr+nv])
    np.savez_compressed(os.path.join(out_dir, 'test.npz'), x=xs[-nt:], y=ys[-nt:])
    # adj pickle
    sensor_ids = [str(i) for i in range(adj.shape[0])]
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    pickle.dump((sensor_ids, sensor_id_to_ind, adj),
                open(os.path.join(out_dir, 'adj_mx.pkl'), 'wb'))
    print(f"SYNTH data: {n} samples, {adj.shape[0]} nodes -> {out_dir}")
    print(f"  train={nr} val={nv} test={nt}")

if __name__ == '__main__':
    data, adj = generate_synth()
    out_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'SYNTH')
    make_splits(data, adj, out_dir)
    print("Done.")
