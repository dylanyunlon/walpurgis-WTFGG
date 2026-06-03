import pickle
import numpy as np

# Delta vs upstream:
#   1. Edge counting uses vectorised np.count_nonzero instead of double loop
#   2. Prints degree distribution stats (min/max/mean/median) per graph


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def describe(name, adj_mx):
    # ── delta 1: vectorised count ──
    edge = int(np.count_nonzero(adj_mx))
    n = adj_mx.shape[0]
    print(f"==================== {name} ====================")
    print(f"# Node: {n}")
    print(f"# Edge: {edge}")
    print(f"  density: {edge / (n * n) * 100:.2f}%")
    # ── delta 2: degree stats ──
    deg = np.count_nonzero(adj_mx, axis=1)
    print(f"  degree  min={deg.min()} max={deg.max()} "
          f"mean={deg.mean():.1f} median={np.median(deg):.1f}")
    print()


file_path = "datasets/sensor_graph/adj_mx_la.pkl"
adj_mx = load_pickle(file_path)[2]
describe("METR-LA", adj_mx)

file_path = "datasets/sensor_graph/adj_mx_bay.pkl"
adj_mx = load_pickle(file_path)[2]
describe("PEMS-BAY", adj_mx)

file_path = "datasets/sensor_graph/adj_mx_04.pkl"
adj_mx = load_pickle(file_path)
describe("PEMS04", adj_mx)

file_path = "datasets/sensor_graph/adj_mx_08.pkl"
adj_mx = load_pickle(file_path)
describe("PEMS08", adj_mx)
