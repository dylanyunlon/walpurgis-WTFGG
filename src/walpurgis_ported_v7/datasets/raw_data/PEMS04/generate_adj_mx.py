import numpy as np
import csv
import pickle
import sys

_DBG_ADJGEN = ("--dbg-adjgen" in sys.argv)


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split('\n'))}
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices,
                                    id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split('\n'))}
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def gaussian_weight_adj(distance_mx, sigma_sq=None):
    """算法改动: 从 binary adj 改为 distance-based Gaussian weighting
    W_ij = exp(-d_ij^2 / sigma^2), 其中 sigma 默认用距离标准差
    非零边才参与计算; 零距离 (自环或无边) 保持为 0"""
    mask = (distance_mx > 0).astype(np.float32)
    nonzero_dists = distance_mx[distance_mx > 0]
    if len(nonzero_dists) == 0:
        return distance_mx
    if sigma_sq is None:
        sigma_sq = nonzero_dists.std() ** 2
        if sigma_sq < 1e-8:
            sigma_sq = 1.0
    W = np.exp(-distance_mx ** 2 / sigma_sq) * mask
    if _DBG_ADJGEN:
        print(f"[DBG-ADJGEN] gaussian_weight  sigma^2={sigma_sq:.4f}  "
              f"W_range=[{W[W > 0].min():.4f}, {W.max():.4f}]  "
              f"edges={mask.sum():.0f}")
    return W


if __name__ == "__main__":
    direction = True
    distance_df_filename = "datasets/raw_data/PEMS04/PEMS04.csv"
    num_of_vertices = 307
    id_filename = None

    if direction:
        adj_mx, distance_mx = get_adjacency_matrix_2direction(
            distance_df_filename, num_of_vertices, id_filename=None)
    else:
        adj_mx, distance_mx = get_adjacency_matrix(
            distance_df_filename, num_of_vertices, id_filename=None)

    add_self_loop = False
    if add_self_loop:
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])

    # 算法改动: 生成 Gaussian weighted adj 作为额外输出
    gauss_adj = gaussian_weight_adj(distance_mx)

    if _DBG_ADJGEN:
        print(f"[DBG-ADJGEN] adj_mx  shape={adj_mx.shape}  "
              f"edges={adj_mx.sum():.0f}  "
              f"density={adj_mx.mean():.4f}")
        print(f"[DBG-ADJGEN] gauss_adj  mean_weight="
              f"{gauss_adj[gauss_adj > 0].mean():.4f}")

    pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
    pickle.dump(distance_mx, open(
        "datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))
    pickle.dump(gauss_adj, open(
        "datasets/sensor_graph/adj_mx_04_gauss.pkl", 'wb'))
