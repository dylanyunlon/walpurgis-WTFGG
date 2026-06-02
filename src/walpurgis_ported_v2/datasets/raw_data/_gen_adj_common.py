"""
Build adjacency matrix from CSV distance files (PEMS04, PEMS08).
Supports both unidirectional and bidirectional graph construction,
plus .npy shortcut for pre-computed matrices.
"""

import csv
import pickle
import sys

import numpy as np

_DBG_ADJ = ("--debug-adj" in sys.argv) or False


def _build_adj_from_csv(csv_path, n_nodes, id_file=None, bidirectional=False):
    """
    Read a CSV distance file and return (adj_binary, adj_distance).

    Parameters
    ----------
    csv_path      : path to CSV with columns [from, to, distance]
    n_nodes       : number of vertices
    id_file       : optional file mapping raw IDs → 0-based indices
    bidirectional : if True, add edges in both directions
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    # optional ID remapping
    id_map = None
    if id_file is not None:
        with open(id_file, 'r') as f:
            raw_ids = f.read().strip().split('\n')
        id_map = {int(raw): idx for idx, raw in enumerate(raw_ids)}

    with open(csv_path, 'r') as f:
        f.readline()                        # skip header
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            src, dst, d = int(row[0]), int(row[1]), float(row[2])
            if id_map:
                src, dst = id_map[src], id_map[dst]
            adj[src, dst] = 1.0
            dist[src, dst] = d
            if bidirectional:
                adj[dst, src] = 1.0
                dist[dst, src] = d

    if _DBG_ADJ:
        n_edges = int(adj.sum())
        print(f"[DBG:adj_gen] csv={csv_path}  nodes={n_nodes}  "
              f"edges={n_edges}  bidir={bidirectional}")
    return adj, dist


def build_adj(csv_path, n_nodes, id_file=None, bidirectional=True):
    """
    High-level entry: handles .npy shortcut or delegates to CSV reader.
    """
    if csv_path.endswith('.npy'):
        mx = np.load(csv_path)
        if _DBG_ADJ:
            print(f"[DBG:adj_gen] loaded npy  shape={mx.shape}")
        return mx, None
    return _build_adj_from_csv(csv_path, n_nodes, id_file, bidirectional)


def save_adj(adj, dist, adj_out, dist_out):
    """Pickle-dump adjacency and distance matrices."""
    pickle.dump(adj,  open(adj_out,  'wb'))
    pickle.dump(dist, open(dist_out, 'wb'))
    print(f"Saved  adj → {adj_out}  dist → {dist_out}")
