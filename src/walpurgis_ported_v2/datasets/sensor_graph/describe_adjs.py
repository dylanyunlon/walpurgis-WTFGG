"""
Utility script: print node/edge counts for all sensor graph adjacency files.
"""

import pickle
import sys

_DBG = ("--debug-adj" in sys.argv) or False


def _load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Cannot load {path}: {e}')
        raise


def _count_edges(adj):
    """Count non-zero entries in a dense adjacency matrix."""
    n_edges = 0
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                n_edges += 1
    return n_edges


def describe(name, path, idx=None):
    """Load and print summary for one adjacency file."""
    raw = _load_pickle(path)
    adj = raw[idx] if idx is not None else raw
    n_edges = _count_edges(adj)
    print(f"{'='*20} {name} {'='*20}")
    print(f"  Nodes: {adj.shape[0]}")
    print(f"  Edges: {n_edges}")
    if _DBG:
        print(f"  [DBG] adj shape={adj.shape}  dtype={adj.dtype}  "
              f"density={n_edges/(adj.shape[0]**2):.4f}")


if __name__ == '__main__':
    describe("METR-LA",  "datasets/sensor_graph/adj_mx_la.pkl",  idx=2)
    describe("PEMS-BAY", "datasets/sensor_graph/adj_mx_bay.pkl", idx=2)
    describe("PEMS04",   "datasets/sensor_graph/adj_mx_04.pkl")
    describe("PEMS08",   "datasets/sensor_graph/adj_mx_08.pkl")
