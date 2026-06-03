"""Graph adjacency inspector — sparse-path rewrite.

Upstream uses O(N^2) double-loop to count edges; this version converts to
CSR once and reads ``nnz`` directly, which is orders of magnitude faster
on large graphs (PEMS-BAY has 325 nodes → 105k cells).
Also prints density, max/min nonzero weight, and degree histogram bins
so you get a real picture of graph structure at a glance.
"""

import pickle
import numpy as np
from scipy import sparse as sp


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as fh:
            return pickle.load(fh)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as fh:
            return pickle.load(fh, encoding='latin1')


def inspect_adj(name, adj_mx):
    csr = sp.csr_matrix(adj_mx)
    n_nodes = csr.shape[0]
    n_edges = csr.nnz
    density = n_edges / (n_nodes * n_nodes)
    weights = csr.data
    degrees = np.diff(csr.indptr)          # out-degree per node

    print(f"{'=' * 20} {name} {'=' * 20}")
    print(f"  Nodes : {n_nodes}")
    print(f"  Edges : {n_edges}  (density {density:.4f})")
    if len(weights) > 0:
        print(f"  Weight: min={weights.min():.6f}  max={weights.max():.6f}  "
              f"mean={weights.mean():.6f}")
    print(f"  Degree: min={degrees.min()}  max={degrees.max()}  "
          f"mean={degrees.mean():.1f}  median={np.median(degrees):.0f}")
    # 5-bin degree histogram for quick structural fingerprint
    bins = np.linspace(degrees.min(), degrees.max(), 6)
    hist, _ = np.histogram(degrees, bins=bins)
    print(f"  DegreeHist(5bin): {hist.tolist()}")


if __name__ == "__main__":
    specs = [
        ("METR-LA",  "datasets/sensor_graph/adj_mx_la.pkl",  True),
        ("PEMS-BAY", "datasets/sensor_graph/adj_mx_bay.pkl", True),
        ("PEMS04",   "datasets/sensor_graph/adj_mx_04.pkl",  False),
        ("PEMS08",   "datasets/sensor_graph/adj_mx_08.pkl",  False),
    ]
    for tag, path, has_ids in specs:
        raw = load_pickle(path)
        mx = raw[2] if has_ids else raw
        inspect_adj(tag, mx)
