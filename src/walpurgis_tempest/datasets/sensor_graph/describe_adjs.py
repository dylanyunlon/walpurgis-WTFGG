"""Describe adjacency matrices."""
import pickle, sys, numpy as np, os

def describe(adj_path):
    adj = pickle.load(open(adj_path, 'rb'))
    if isinstance(adj, tuple): adj = adj[2]
    a = np.array(adj)
    print(f"Shape: {a.shape}, dtype: {a.dtype}")
    print(f"Range: [{a.min():.4f}, {a.max():.4f}], Mean: {a.mean():.4f}")
    print(f"Density: {(a > 0).sum() / a.size:.2%}, Symmetric: {np.allclose(a, a.T)}")

if __name__ == '__main__':
    if len(sys.argv) > 1: describe(sys.argv[1])
    else: print("Usage: python describe_adjs.py <adj_path.pkl>")
