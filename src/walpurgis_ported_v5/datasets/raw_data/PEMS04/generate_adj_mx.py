import numpy as np
import csv
import pickle

# Delta vs upstream:
#   1. Removed duplicate function definitions (upstream had 3 copies)
#   2. Prints adjacency sparsity after construction


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
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


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
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


direction = True
distance_df_filename, num_of_vertices, id_filename = "datasets/raw_data/PEMS04/PEMS04.csv", 307, None
if direction:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None)
else:
    adj_mx, distance_mx = get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None)

add_self_loop = False
if add_self_loop:
    adj_mx = adj_mx + np.identity(adj_mx.shape[0])
    distance_mx = distance_mx + np.identity(distance_mx.shape[0])

# ── delta 2: sparsity summary ──
nnz = int(np.count_nonzero(adj_mx))
total = adj_mx.shape[0] * adj_mx.shape[1]
print(f"PEMS04 adj: {nnz}/{total} non-zero ({nnz/total*100:.2f}%)")

pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))
