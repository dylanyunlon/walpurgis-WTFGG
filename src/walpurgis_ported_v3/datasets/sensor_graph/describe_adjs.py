"""Quick-inspect adjacency matrices for all four datasets."""
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


def _report(tag, adj):
    print(f"{'='*20} {tag} {'='*20}")
    print(f"  Nodes: {adj.shape[0]}")
    print(f"  Edges: {_count_edges(adj)}")


if __name__ == "__main__":
    # METR-LA
    obj = _load_pkl("datasets/sensor_graph/adj_mx_la.pkl")
    _report("METR-LA", obj[2])

    # PEMS-BAY
    obj = _load_pkl("datasets/sensor_graph/adj_mx_bay.pkl")
    _report("PEMS-BAY", obj[2])

    # PEMS04
    _report("PEMS04", _load_pkl("datasets/sensor_graph/adj_mx_04.pkl"))

    # PEMS08
    _report("PEMS08", _load_pkl("datasets/sensor_graph/adj_mx_08.pkl"))
