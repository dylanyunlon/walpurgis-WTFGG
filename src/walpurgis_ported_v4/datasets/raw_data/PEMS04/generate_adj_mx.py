"""
Walpurgis V4 — PEMS04 Adjacency Matrix Generator
Upstream: datasets/raw_data/PEMS04/generate_adj_mx.py

Modifications (~20%):
  1. Removed duplicate get_adjacency_matrix_2direction definitions (upstream has 3 copies!)
  2. Unified into single build_adjacency(path, n, direction, id_file) function
  3. Added validation: checks for self-loops, isolated nodes, prints graph stats
  4. Auto-creates output directory
"""

import csv
import os
import pickle
import sys

import numpy as np


def build_adjacency(distance_df_filename, num_of_vertices, direction=True, id_filename=None):
    """Build adjacency and distance matrices from CSV edge list.

    Args:
        distance_df_filename: path to CSV with (from, to, distance) rows
        num_of_vertices: number of nodes
        direction: if True, make edges bidirectional
        id_filename: optional file mapping raw IDs to 0-indexed

    Returns:
        A: binary adjacency matrix (N, N)
        distanceA: distance-weighted matrix (N, N)
    """
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None

    N = int(num_of_vertices)
    A = np.zeros((N, N), dtype=np.float32)
    distanceA = np.zeros((N, N), dtype=np.float32)

    id_dict = None
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

    with open(distance_df_filename, 'r') as f:
        f.readline()  # skip header
        reader = csv.reader(f)
        edge_count = 0
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if id_dict:
                i, j = id_dict[i], id_dict[j]

            A[i, j] = 1
            distanceA[i, j] = distance

            if direction:
                A[j, i] = 1
                distanceA[j, i] = distance

            edge_count += 1

    # V4: validation and diagnostics
    n_edges     = int(np.count_nonzero(A))
    self_loops  = int(np.trace(A))
    isolated    = int(np.sum(A.sum(axis=1) == 0))
    density     = n_edges / (N * N) * 100

    print(f"[V4-AdjMx] PEMS04 adjacency built:", file=sys.stderr)
    print(f"  Nodes: {N}, Edges: {n_edges} (from {edge_count} CSV rows, direction={direction})", file=sys.stderr)
    print(f"  Self-loops: {self_loops}, Isolated nodes: {isolated}", file=sys.stderr)
    print(f"  Density: {density:.2f}%", file=sys.stderr)
    print(f"  Symmetric: {np.allclose(A, A.T)}", file=sys.stderr)

    if isolated > 0:
        print(f"  [WARN] {isolated} isolated nodes detected!", file=sys.stderr)

    return A, distanceA


if __name__ == "__main__":
    distance_df_filename = "datasets/raw_data/PEMS04/PEMS04.csv"
    num_of_vertices = 307
    direction = True
    add_self_loop = False

    adj_mx, distance_mx = build_adjacency(
        distance_df_filename, num_of_vertices,
        direction=direction, id_filename=None
    )

    if add_self_loop:
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])

    os.makedirs("datasets/sensor_graph", exist_ok=True)
    pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
    pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))
    print("[V4] Saved adj_mx_04.pkl and adj_mx_04_distance.pkl", file=sys.stderr)
