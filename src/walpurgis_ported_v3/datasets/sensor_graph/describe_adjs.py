"""
Quick summary of all adjacency matrices: node count and edge count.
"""
import pickle


def _load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def _count_edges(adj):
    return int(sum(1 for i in range(adj.shape[0])
                   for j in range(adj.shape[1])
                   if adj[i][j] != 0))


def _describe(tag, path, unpack_idx=None):
    raw = _load_pkl(path)
    mx = raw[unpack_idx] if unpack_idx is not None else raw
    n_edges = _count_edges(mx)
    print(f"{'='*20} {tag} {'='*20}")
    print(f"  Nodes: {mx.shape[0]}    Edges: {n_edges}")


if __name__ == "__main__":
    _describe("METR-LA",  "datasets/sensor_graph/adj_mx_la.pkl",  unpack_idx=2)
    _describe("PEMS-BAY", "datasets/sensor_graph/adj_mx_bay.pkl", unpack_idx=2)
    _describe("PEMS04",   "datasets/sensor_graph/adj_mx_04.pkl")
    _describe("PEMS08",   "datasets/sensor_graph/adj_mx_08.pkl")
