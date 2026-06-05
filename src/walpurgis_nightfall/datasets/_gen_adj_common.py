"""
_gen_adj_common — Nightfall变体
PEMS04和PEMS08共用的邻接矩阵生成
算法改写:
  1. 读取后自动验证连通性 (BFS连通分量)
  2. 对称化后验证对称性
  3. 稀疏度打印
"""
import numpy as np
import csv
import pickle
import sys


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distA[id_dict[i], id_dict[j]] = distance
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distA[i, j] = distance
    return A, distA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
                distA[id_dict[i], id_dict[j]] = distance
                distA[id_dict[j], id_dict[i]] = distance
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
                distA[i, j] = distance
                distA[j, i] = distance
    return A, distA


def _check_connectivity(adj_mx):
    n = adj_mx.shape[0]
    binary = adj_mx > 0
    visited = set()
    queue = [0]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        neighbors = np.where(binary[node] | binary[:, node])[0]
        queue.extend(int(nb) for nb in neighbors if nb not in visited)
    components = 1 if len(visited) == n else "DISCONNECTED"
    density = (adj_mx > 0).sum() / (n * n)
    print(f"[NF-ADJ] nodes={n} edges={(adj_mx>0).sum()} density={density:.4f} connected={components}")
    return len(visited) == n


def build_and_save(csv_path, num_vertices, adj_out, dist_out,
                   direction=True, add_self_loop=False, id_filename=None):
    if direction:
        adj_mx, dist_mx = get_adjacency_matrix_2direction(csv_path, num_vertices, id_filename)
    else:
        adj_mx, dist_mx = get_adjacency_matrix(csv_path, num_vertices, id_filename)
    if add_self_loop:
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        dist_mx = dist_mx + np.identity(dist_mx.shape[0])
    # 对称性验证
    assert np.allclose(adj_mx, adj_mx.T), "Adjacency matrix is not symmetric after 2-direction build!"
    _check_connectivity(adj_mx)
    pickle.dump(adj_mx, open(adj_out, 'wb'))
    pickle.dump(dist_mx, open(dist_out, 'wb'))
    print(f"[NF-ADJ] Saved {adj_out} and {dist_out}")
