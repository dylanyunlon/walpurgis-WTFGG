"""
generate_adj_mx.py (PEMS08) — v9 port
Algo delta (same as PEMS04):
  1. Gaussian kernel weighted adjacency
  2. k-NN sparsification (k=20)
  3. symmetric closure
"""
import numpy as np
import csv
import pickle

_K_NN = 20


def get_adjacency_matrix_gaussian(distance_df_filename, num_of_vertices, id_filename=None, k_nn=_K_NN):
    if 'npy' in distance_df_filename:
        return np.load(distance_df_filename), None

    distA = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
    else:
        id_dict = None

    with open(distance_df_filename, 'r') as f:
        f.readline()
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if id_dict:
                i, j = id_dict[i], id_dict[j]
            distA[i, j] = d
            distA[j, i] = d

    nonzero = distA[distA > 0]
    sigma_sq = np.mean(nonzero ** 2) if len(nonzero) > 0 else 1.0
    W = np.where(distA > 0, np.exp(-distA**2 / sigma_sq), 0.0).astype(np.float32)

    for i in range(num_of_vertices):
        row = W[i, :]
        if np.count_nonzero(row) > k_nn:
            threshold = np.sort(row)[::-1][k_nn]
            W[i, row < threshold] = 0.0

    W = np.maximum(W, W.T)
    print(f"v9 PEMS08  σ²={sigma_sq:.4f}  edges={(W>0).sum()}  density={(W>0).mean()*100:.2f}%")
    return W, distA


distance_df_filename = "datasets/raw_data/PEMS08/PEMS08.csv"
num_of_vertices = 170

adj_mx, distance_mx = get_adjacency_matrix_gaussian(distance_df_filename, num_of_vertices)
pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_08.pkl", 'wb'))
pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_08_distance.pkl", 'wb'))
