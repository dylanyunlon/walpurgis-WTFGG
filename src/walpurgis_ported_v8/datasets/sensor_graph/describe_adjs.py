import pickle
import numpy as np


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


def describe_adj(file_path, name, has_tuple=True):
    raw = load_pickle(file_path)
    adj_mx = raw[2] if has_tuple else raw
    edge = 0
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if adj_mx[i][j] != 0:
                edge += 1
    # degree distribution
    degrees = np.sum(adj_mx > 0, axis=1)
    print(f"==================== {name} ====================")
    print(f"# Node: {adj_mx.shape[0]}")
    print(f"# Edge: {edge}")
    print(f"  Avg degree: {degrees.mean():.2f}")
    print(f"  Max degree: {degrees.max()}")
    print(f"  Min degree: {degrees.min()}")
    print(f"  Density: {edge / (adj_mx.shape[0] ** 2):.6f}")


describe_adj("datasets/sensor_graph/adj_mx_la.pkl", "METR-LA", True)
describe_adj("datasets/sensor_graph/adj_mx_bay.pkl", "PEMS-BAY", True)
describe_adj("datasets/sensor_graph/adj_mx_04.pkl", "PEMS04", False)
describe_adj("datasets/sensor_graph/adj_mx_08.pkl", "PEMS08", False)
