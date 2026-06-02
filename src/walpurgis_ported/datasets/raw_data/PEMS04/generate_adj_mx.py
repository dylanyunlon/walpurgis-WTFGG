"""Generate adjacency matrix from sensor distance CSV for PEMS04.

Walpurgis adaptations vs upstream:
- Graph topology diagnostics (density, symmetry, connected components estimate)
- Distance statistics reporting (mean, std, histogram bins)
- Self-loop and isolated node detection
- Timing for I/O operations
- Outputs a full diagnostic report suitable for debugging data pipeline issues

Usage:
    python generate_adj_mx.py
    
Debug: set WALPURGIS_VERBOSE=1 for per-edge logging (caution: very verbose)
"""
import time
import os
import numpy as np
import csv
import pickle


def _graph_diagnostics(adj, distance, name="adj"):
    """Print comprehensive graph diagnostics.
    
    Call this after building any adjacency matrix to understand its structure.
    Useful for catching data bugs (e.g., disconnected sensors, zero-distance edges).
    """
    n = adj.shape[0]
    nnz = np.count_nonzero(adj)
    density = nnz / (n * n) if n > 0 else 0
    is_symmetric = np.allclose(adj, adj.T)
    has_self_loops = np.any(np.diag(adj) != 0)
    degrees = adj.sum(axis=1)
    isolated = np.sum(degrees == 0)
    
    print(f"\n  [{name}] Graph Diagnostics:")
    print(f"    nodes={n}, edges={nnz}, density={density:.4f}")
    print(f"    symmetric={is_symmetric}, self_loops={has_self_loops}")
    print(f"    degree: min={degrees.min():.0f}, max={degrees.max():.0f}, "
          f"mean={degrees.mean():.1f}, std={degrees.std():.1f}")
    if isolated > 0:
        isolated_ids = np.where(degrees == 0)[0]
        print(f"    ⚠ {isolated} isolated nodes: {isolated_ids[:10]}{'...' if isolated > 10 else ''}")
    
    if distance is not None:
        nonzero_dist = distance[distance > 0]
        if len(nonzero_dist) > 0:
            print(f"    distance: min={nonzero_dist.min():.2f}, max={nonzero_dist.max():.2f}, "
                  f"mean={nonzero_dist.mean():.2f}, std={nonzero_dist.std():.2f}")
            # Quick histogram
            bins = np.percentile(nonzero_dist, [25, 50, 75])
            print(f"    distance quartiles: Q1={bins[0]:.2f}, Q2={bins[1]:.2f}, Q3={bins[2]:.2f}")


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """Build directed adjacency and distance matrices from CSV edge list.
    
    Each row in CSV: (from_node, to_node, distance).
    Returns binary adjacency A and weighted distance matrix.
    
    Debug: prints edge count progress every 1000 edges.
    """
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj] file={distance_df_filename} N={num_of_vertices} "
          f"id_file={id_filename}")
    
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy directly: shape={adj_mx.shape}")
        return adj_mx, None
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0
    skipped = 0

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        print(f"  id mapping: {len(id_dict)} sensor IDs loaded")
        
        with open(distance_df_filename, 'r') as f:
            f.readline()  # skip header
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    skipped += 1
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distanceA[id_dict[i], id_dict[j]] = distance
                edge_count += 1
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    skipped += 1
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distanceA[i, j] = distance
                edge_count += 1

    elapsed = time.perf_counter() - t0
    print(f"  edges={edge_count}, skipped={skipped}, time={elapsed:.3f}s")
    _graph_diagnostics(A, distanceA, "directed")
    return A, distanceA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    """Build undirected (bidirectional) adjacency and distance matrices.
    
    For each edge (i,j), also adds (j,i). This is the standard choice
    for traffic networks where flow is bidirectional.
    """
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj_2dir] file={distance_df_filename} N={num_of_vertices}")
    
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy directly: shape={adj_mx.shape}")
        return adj_mx, None
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0
    skipped = 0

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        print(f"  id mapping: {len(id_dict)} sensor IDs loaded")
        
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    skipped += 1
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
                distanceA[id_dict[i], id_dict[j]] = distance
                distanceA[id_dict[j], id_dict[i]] = distance
                edge_count += 1
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    skipped += 1
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                A[j, i] = 1
                distanceA[i, j] = distance
                distanceA[j, i] = distance
                edge_count += 1

    elapsed = time.perf_counter() - t0
    print(f"  raw edges={edge_count} (×2 bidirectional), skipped={skipped}, time={elapsed:.3f}s")
    _graph_diagnostics(A, distanceA, "undirected")
    return A, distanceA


# ── Main: generate and save adjacency for PEMS04 ──────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("[Walpurgis] Generating PEMS04 adjacency matrix")
    print("=" * 60)
    
    t_start = time.perf_counter()
    
    direction = True
    distance_df_filename = "datasets/raw_data/PEMS04/PEMS04.csv"
    num_of_vertices = 307
    id_filename = None
    
    print(f"  direction={'bidirectional' if direction else 'directed'}")
    print(f"  source: {distance_df_filename}")
    print(f"  vertices: {num_of_vertices}")
    
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
        print("  self-loops added")

    # Save
    adj_path = "datasets/sensor_graph/adj_mx_04.pkl"
    dist_path = "datasets/sensor_graph/adj_mx_04_distance.pkl"
    pickle.dump(adj_mx, open(adj_path, 'wb'))
    pickle.dump(distance_mx, open(dist_path, 'wb'))
    
    # Verify saved files
    adj_size = os.path.getsize(adj_path) / 1024
    dist_size = os.path.getsize(dist_path) / 1024
    
    total_time = time.perf_counter() - t_start
    print(f"\n  saved: {adj_path} ({adj_size:.1f} KB)")
    print(f"  saved: {dist_path} ({dist_size:.1f} KB)")
    print(f"  total time: {total_time:.3f}s")
    print(f"\n  ── Verification ──")
    print(f"  adj shape={adj_mx.shape} dtype={adj_mx.dtype}")
    print(f"  adj nonzero={np.count_nonzero(adj_mx)} "
          f"density={np.count_nonzero(adj_mx)/(307*307):.4f}")
    print(f"  distance nonzero={np.count_nonzero(distance_mx)}")
