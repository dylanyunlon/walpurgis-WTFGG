"""
Common adjacency matrix builder for PEMS04 / PEMS08.
Reads CSV edge list, outputs pickle.
"""
import numpy as np
import csv
import pickle


def _read_adj_unidirectional(csv_path, n_nodes, id_file=None):
    """Build (A, dist_A) from a one-directional edge CSV."""
    if csv_path.endswith('.npy'):
        return np.load(csv_path), None

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    D = np.zeros_like(A)

    id_map = None
    if id_file:
        with open(id_file) as f:
            id_map = {int(v): idx for idx, v in
                      enumerate(f.read().strip().split('\n'))}

    with open(csv_path) as f:
        f.readline()  # skip header
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if id_map:
                i, j = id_map[i], id_map[j]
            A[i, j] = 1
            D[i, j] = d

    return A, D


def _read_adj_bidirectional(csv_path, n_nodes, id_file=None):
    """Build symmetric (A, dist_A) from edge CSV."""
    if csv_path.endswith('.npy'):
        return np.load(csv_path), None

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    D = np.zeros_like(A)

    id_map = None
    if id_file:
        with open(id_file) as f:
            id_map = {int(v): idx for idx, v in
                      enumerate(f.read().strip().split('\n'))}

    with open(csv_path) as f:
        f.readline()
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if id_map:
                i, j = id_map[i], id_map[j]
            A[i, j] = A[j, i] = 1
            D[i, j] = D[j, i] = d

    return A, D


def build_and_save(csv_path, n_nodes, adj_out, dist_out,
                   bidirectional=True, self_loop=False):
    """End-to-end: read CSV → build adj → optionally add self-loop → pickle."""
    reader = _read_adj_bidirectional if bidirectional else _read_adj_unidirectional
    adj, dist = reader(csv_path, n_nodes)

    if self_loop:
        eye = np.identity(n_nodes)
        adj  = adj  + eye
        dist = dist + eye

    pickle.dump(adj,  open(adj_out,  'wb'))
    pickle.dump(dist, open(dist_out, 'wb'))
    print(f"[adj] saved  adj→{adj_out}  dist→{dist_out}  "
          f"N={n_nodes}  edges={int(adj.sum())}")
