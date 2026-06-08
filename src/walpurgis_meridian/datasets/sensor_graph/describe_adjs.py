"""Describe adjacency matrices for sensor graph datasets."""
import pickle
import numpy as np
import sys


def describe_adj(path):
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(open(path, 'rb'))
        print(f"Sensors: {len(sensor_ids)}")
        print(f"Adj shape: {adj_mx.shape}")
        print(f"Non-zero: {np.count_nonzero(adj_mx)}")
        print(f"Density: {np.count_nonzero(adj_mx) / adj_mx.size:.4f}")
    except Exception:
        adj_mx = pickle.load(open(path, 'rb'))
        if isinstance(adj_mx, np.ndarray):
            print(f"Adj shape: {adj_mx.shape}")
            print(f"Non-zero: {np.count_nonzero(adj_mx)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        describe_adj(sys.argv[1])
