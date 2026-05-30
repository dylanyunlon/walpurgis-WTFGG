"""Generate adjacency matrix from sensor distance CSV for PEMS04.

Walpurgis adaptations:
- Graph topology diagnostics (density, symmetry, connected components estimate)
- Distance statistics reporting
- Self-loop and isolated node detection
- Timing for I/O operations
"""
import time
import numpy as np
import csv
import pickle


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """Build directed adjacency + distance matrices from CSV edge list.

    Walpurgis: reports edge count and sparsity.

    Returns:
        A: binary adjacency [N, N]
        distanceA: weighted adjacency [N, N]
    """
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj_mx] file={distance_df_filename} N={num_of_vertices} "
          f"id_file={id_filename}")

    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy: shape={adj_mx.shape}")
        return adj_mx, None

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        print(f"  id_dict: {len(id_dict)} sensor IDs mapped")
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
                edge_count += 1
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
                edge_count += 1

    elapsed = time.perf_counter() - t0
    nnz = np.count_nonzero(A)
    density = nnz / (num_of_vertices ** 2)
    print(f"  edges={edge_count} nnz={nnz} density={density:.4f} elapsed={elapsed:.3f}s")
    if nnz > 0:
        dists = distaneA[distaneA > 0]
        print(f"  distance stats: mean={dists.mean():.2f} min={dists.min():.2f} "
              f"max={dists.max():.2f} std={dists.std():.2f}")

    return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    """Build undirected (symmetric) adjacency + distance matrices.

    Walpurgis: verifies symmetry after construction.
    """
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj_mx_2dir] file={distance_df_filename} N={num_of_vertices}")

    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy: shape={adj_mx.shape}")
        return adj_mx, None

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0

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
                edge_count += 1
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
                edge_count += 1

    elapsed = time.perf_counter() - t0
    nnz = np.count_nonzero(A)
    density = nnz / (num_of_vertices ** 2)
    is_sym = np.allclose(A, A.T)
    isolated = np.sum(A.sum(axis=1) == 0)
    print(f"  edges_read={edge_count} nnz={nnz} density={density:.4f} "
          f"symmetric={is_sym} isolated_nodes={isolated} elapsed={elapsed:.3f}s")

    return A, distaneA


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"[Walpurgis::PEMS04] Generating adjacency matrix")
    print(f"{'='*60}")

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
        print(f"[Walpurgis] Added self-loops")

    # Save
    pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
    pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))
    print(f"\n[Walpurgis::PEMS04] Saved adj_mx_04.pkl and adj_mx_04_distance.pkl")
    print(f"  adj_mx: {adj_mx.shape} nnz={np.count_nonzero(adj_mx)}")
    print(f"  distance_mx: {distance_mx.shape} "
          f"mean_dist={distance_mx[distance_mx > 0].mean():.2f}" if distance_mx is not None else "")
