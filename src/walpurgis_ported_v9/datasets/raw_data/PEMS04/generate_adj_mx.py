"""
generate_adj_mx.py (PEMS04) — v9 port
Algo delta:
  1. upstream: 二值邻接 A[i,j] = 1 if connected, 0 otherwise
     → v9: Gaussian kernel 加权:
       W[i,j] = exp(-d²/σ²)  (σ² = mean(d²) of all edges)
     保留距离信息, 近邻权重大, 远邻权重小
  2. k-NN 稀疏化: 每个节点只保留 top-k 最大权重的邻居 (k=20)
  3. 对称闭包: A_sym = max(A, A^T), 保证无向
"""
import numpy as np
import csv
import pickle

_K_NN = 20


def get_adjacency_matrix_gaussian(distance_df_filename, num_of_vertices, id_filename=None, k_nn=_K_NN):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None

    distA = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
    else:
        id_dict = None

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if id_dict:
                i, j = id_dict[i], id_dict[j]
            distA[i, j] = d
            distA[j, i] = d  # symmetric

    # v9: Gaussian kernel weighting
    nonzero = distA[distA > 0]
    if len(nonzero) > 0:
        sigma_sq = np.mean(nonzero ** 2)
    else:
        sigma_sq = 1.0
    W = np.where(distA > 0, np.exp(-distA ** 2 / sigma_sq), 0.0).astype(np.float32)
    print(f"v9 Gaussian kernel  σ²={sigma_sq:.4f}  nonzero_edges={len(nonzero)}")

    # v9: k-NN sparsification
    for i in range(num_of_vertices):
        row = W[i, :]
        if np.count_nonzero(row) > k_nn:
            threshold = np.sort(row)[::-1][k_nn]
            W[i, row < threshold] = 0.0

    # v9: symmetric closure
    W = np.maximum(W, W.T)
    print(f"v9 k-NN(k={k_nn})  final_edges={(W > 0).sum()}")

    return W, distA


# ── main ──
distance_df_filename = "datasets/raw_data/PEMS04/PEMS04.csv"
num_of_vertices = 307

adj_mx, distance_mx = get_adjacency_matrix_gaussian(
    distance_df_filename, num_of_vertices, id_filename=None)

pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))
print(f"saved  adj_mx shape={adj_mx.shape}  density={(adj_mx>0).mean()*100:.2f}%")
