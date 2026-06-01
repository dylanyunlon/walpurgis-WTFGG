"""Generate adjacency matrix from sensor distance CSV for PEMS08.

Walpurgis adaptations vs upstream:
- Graph topology diagnostics (density, symmetry, connected components estimate)
- Distance statistics reporting (mean, std, quartiles)
- Self-loop and isolated node detection
- Timing for I/O operations
- PEMS08 is the smallest dataset (170 nodes) — serves as canary for pipeline validation

Usage:
    python generate_adj_mx.py

Debug: compare output diagnostics with PEMS04 to ensure consistency.
PEMS08 should have similar density but fewer nodes.
"""
import time
import os
import numpy as np
import csv
import pickle


def _graph_diagnostics(adj, distance, name="adj"):
    """Print comprehensive graph diagnostics.
    
    Call after building any adjacency matrix to verify its structure.
    For PEMS08 (170 nodes), this runs near-instantly — no need to skip.
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
            bins = np.percentile(nonzero_dist, [25, 50, 75])
            print(f"    distance quartiles: Q1={bins[0]:.2f}, Q2={bins[1]:.2f}, Q3={bins[2]:.2f}")
    
    # PEMS08-specific: with 170 nodes, we can afford full eigenvalue check
    if n <= 200 and is_symmetric:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            adj_sp = csr_matrix(adj)
            if nnz > 0:
                eig_max = eigsh(adj_sp.astype(float), 1, which='LM', return_eigenvectors=False)[0]
                print(f"    spectral radius (λ_max) ≈ {eig_max:.4f}")
        except Exception as e:
            print(f"    spectral analysis skipped: {e}")


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """Build directed adjacency and distance matrices from CSV edge list."""
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj] file={distance_df_filename} N={num_of_vertices}")
    
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy: shape={adj_mx.shape}")
        return adj_mx, None
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0
    skipped = 0

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        print(f"  id mapping: {len(id_dict)} sensor IDs")
        with open(distance_df_filename, 'r') as f:
            f.readline()
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
    """Build undirected adjacency and distance matrices (bidirectional edges)."""
    t0 = time.perf_counter()
    print(f"[Walpurgis::get_adj_2dir] file={distance_df_filename} N={num_of_vertices}")
    
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        print(f"  loaded .npy: shape={adj_mx.shape}")
        return adj_mx, None
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    edge_count = 0
    skipped = 0

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        print(f"  id mapping: {len(id_dict)} sensor IDs")
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


# ── Main: generate and save adjacency for PEMS08 ──────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("[Walpurgis] Generating PEMS08 adjacency matrix (canary dataset)")
    print("=" * 60)
    
    t_start = time.perf_counter()
    
    direction = True
    distance_df_filename = "datasets/raw_data/PEMS08/PEMS08.csv"
    num_of_vertices = 170
    id_filename = None
    
    print(f"  direction={'bidirectional' if direction else 'directed'}")
    print(f"  source: {distance_df_filename}")
    print(f"  vertices: {num_of_vertices} (smallest in suite)")
    
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

    adj_path = "datasets/sensor_graph/adj_mx_08.pkl"
    dist_path = "datasets/sensor_graph/adj_mx_08_distance.pkl"
    pickle.dump(adj_mx, open(adj_path, 'wb'))
    pickle.dump(distance_mx, open(dist_path, 'wb'))
    
    adj_size = os.path.getsize(adj_path) / 1024
    dist_size = os.path.getsize(dist_path) / 1024
    
    total_time = time.perf_counter() - t_start
    print(f"\n  saved: {adj_path} ({adj_size:.1f} KB)")
    print(f"  saved: {dist_path} ({dist_size:.1f} KB)")
    print(f"  total time: {total_time:.3f}s")
    print(f"\n  ── Verification ──")
    print(f"  adj shape={adj_mx.shape} dtype={adj_mx.dtype}")
    print(f"  adj nonzero={np.count_nonzero(adj_mx)} "
          f"density={np.count_nonzero(adj_mx)/(170*170):.4f}")
    
    # Cross-check: PEMS08 should have similar density to PEMS04
    print(f"\n  ── Canary check ──")
    print(f"  If density differs wildly from PEMS04 (~0.05-0.10), investigate data source.")
