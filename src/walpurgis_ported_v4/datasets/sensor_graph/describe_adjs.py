import pickle
import numpy as np
import sys

_V4_DEBUG = True


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def describe_adj(name, adj_mx):
    """v4: vectorized edge counting + graph statistics (replaces nested loop)"""
    # upstream used O(N^2) nested loop; v4 uses np.count_nonzero — O(1) sparse or O(N^2) dense but no Python loop
    edge = np.count_nonzero(adj_mx)

    print(f"==================== {name} ====================")
    print(f"# Node: {adj_mx.shape[0]}")
    print(f"# Edge: {edge}")

    if _V4_DEBUG:
        density = edge / (adj_mx.shape[0] * adj_mx.shape[1]) if adj_mx.size > 0 else 0
        # v4: compute degree distribution stats
        degree = np.count_nonzero(adj_mx, axis=1)
        nz_vals = adj_mx[adj_mx != 0]
        print(f"  Density: {density:.6f}", file=sys.stderr)
        print(f"  Degree: min={degree.min()} max={degree.max()} "
              f"mean={degree.mean():.2f} std={degree.std():.2f}", file=sys.stderr)
        if len(nz_vals) > 0:
            print(f"  Edge weights: min={nz_vals.min():.6f} max={nz_vals.max():.6f} "
                  f"mean={nz_vals.mean():.6f}", file=sys.stderr)
        # v4: symmetry check
        if adj_mx.shape[0] == adj_mx.shape[1]:
            sym_diff = np.abs(adj_mx - adj_mx.T).sum()
            print(f"  Symmetry error (||A-A^T||_1): {sym_diff:.6f}", file=sys.stderr)


# METR-LA
file_path = "datasets/sensor_graph/adj_mx_la.pkl"
adj_mx = load_pickle(file_path)[2]
describe_adj("METR-LA", adj_mx)

# PEMS-BAY
file_path = "datasets/sensor_graph/adj_mx_bay.pkl"
adj_mx = load_pickle(file_path)[2]
describe_adj("PEMS-BAY", adj_mx)

# PEMS04
file_path = "datasets/sensor_graph/adj_mx_04.pkl"
adj_mx = load_pickle(file_path)
describe_adj("PEMS04", adj_mx)

# PEMS08
file_path = "datasets/sensor_graph/adj_mx_08.pkl"
adj_mx = load_pickle(file_path)
describe_adj("PEMS08", adj_mx)
