"""Common adj generation."""
import numpy as np, pickle, os

def generate_adj_from_distance(dist_path, output_path, num_nodes, sigma=0.1, threshold=0.1):
    print(f"[TEM] Generate adj: {dist_path} -> {output_path}")
    if os.path.exists(dist_path):
        dist = np.load(dist_path)
    else:
        dist = np.random.rand(num_nodes, num_nodes).astype(np.float32)
        dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0)
    adj = np.exp(-dist**2 / (2 * sigma**2)); adj[adj < threshold] = 0; np.fill_diagonal(adj, 0)
    pickle.dump(adj, open(output_path, 'wb'))
    print(f"  adj: {adj.shape} density={adj[adj>0].size/adj.size:.2%}")
