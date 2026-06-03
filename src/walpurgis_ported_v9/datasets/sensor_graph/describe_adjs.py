"""
describe_adjs.py — v9 port
Algo delta:
  1. upstream: O(N²) 双重循环数非零边
     → v9: 转 scipy.sparse.csr_matrix, 直接 .nnz 获得, O(N)
  2. 新增: 谱半径 ρ(A) = max|λ|, 用 ARPACK 稀疏特征值求解
     ρ < 1 → 图扩散 (power iteration) 收敛
     ρ ≥ 1 → 需要归一化后才能做多阶扩散
"""
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def describe(name, file_path, is_tuple=False):
    data = load_pickle(file_path)
    adj_mx = data[2] if is_tuple else data

    sp = csr_matrix(adj_mx)
    n_nodes = adj_mx.shape[0]
    n_edges = sp.nnz

    # v9: spectral radius via ARPACK
    try:
        evals, _ = eigs(sp.astype(np.float64), k=1, which='LM')
        spec_radius = float(np.abs(evals[0]))
    except Exception:
        spec_radius = float('nan')

    density = n_edges / (n_nodes * n_nodes) * 100
    avg_degree = n_edges / n_nodes

    print(f"{'='*20} {name} {'='*20}")
    print(f"  Nodes:           {n_nodes}")
    print(f"  Edges (nnz):     {n_edges}")
    print(f"  Density:         {density:.2f}%")
    print(f"  Avg degree:      {avg_degree:.1f}")
    print(f"  Spectral radius: {spec_radius:.4f}")
    print(f"  Weight range:    [{adj_mx[adj_mx>0].min():.4f}, {adj_mx.max():.4f}]" if n_edges > 0 else "")


describe("METR-LA",  "datasets/sensor_graph/adj_mx_la.pkl",  is_tuple=True)
describe("PEMS-BAY", "datasets/sensor_graph/adj_mx_bay.pkl", is_tuple=True)
describe("PEMS04",   "datasets/sensor_graph/adj_mx_04.pkl",  is_tuple=False)
describe("PEMS08",   "datasets/sensor_graph/adj_mx_08.pkl",  is_tuple=False)
