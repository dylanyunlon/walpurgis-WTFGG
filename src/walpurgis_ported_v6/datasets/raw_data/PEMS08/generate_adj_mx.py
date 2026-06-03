"""PEMS08 adjacency matrix generator — Gaussian kernel + k-NN sparsification.

Same algorithmic changes as PEMS04 version:
1. Gaussian kernel weighting (adaptive σ = median distance).
2. k-NN sparsification (top-20) + symmetric closure.
"""

import numpy as np
import csv
import pickle


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         id_filename=None):
    if 'npy' in distance_df_filename:
        return np.load(distance_df_filename), None

    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    distA = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            for row in csv.reader(f):
                if len(row) != 3:
                    continue
                i, j, d = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
                distA[id_dict[i], id_dict[j]] = d
                distA[id_dict[j], id_dict[i]] = d
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            for row in csv.reader(f):
                if len(row) != 3:
                    continue
                i, j, d = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                A[j, i] = 1
                distA[i, j] = d
                distA[j, i] = d
    return A, distA


def _gaussian_kernel(dist_mx, sigma=None):
    nonzero = dist_mx[dist_mx > 0]
    if sigma is None:
        sigma = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
    print(f"  [gaussian] sigma={sigma:.2f}")
    weights = np.zeros_like(dist_mx)
    mask = dist_mx > 0
    weights[mask] = np.exp(-(dist_mx[mask] ** 2) / (sigma ** 2))
    return weights


def _knn_sparsify(adj, k=20):
    n = adj.shape[0]
    sparse = np.zeros_like(adj)
    for i in range(n):
        row = adj[i]
        if np.count_nonzero(row) <= k:
            sparse[i] = row
        else:
            top_k_idx = np.argsort(row)[-k:]
            sparse[i, top_k_idx] = row[top_k_idx]
    sparse = np.maximum(sparse, sparse.T)
    print(f"  [knn] k={k}  edges {np.count_nonzero(adj)} → "
          f"{np.count_nonzero(sparse)}")
    return sparse


if __name__ == "__main__":
    distance_df_filename = "datasets/raw_data/PEMS08/PEMS08.csv"
    num_of_vertices = 170

    _, dist_mx = get_adjacency_matrix(
        distance_df_filename, num_of_vertices, id_filename=None)

    adj_mx = _gaussian_kernel(dist_mx)
    adj_mx = _knn_sparsify(adj_mx, k=20)

    print(f"  Final adj: shape={adj_mx.shape}  "
          f"nnz={np.count_nonzero(adj_mx)}")

    pickle.dump(adj_mx,
                open("datasets/sensor_graph/adj_mx_08.pkl", 'wb'))
    pickle.dump(dist_mx,
                open("datasets/sensor_graph/adj_mx_08_distance.pkl", 'wb'))
