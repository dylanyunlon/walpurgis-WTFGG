import pickle
import numpy as np
import sys

_DBG_DESC = ("--dbg-desc" in sys.argv)


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


def _graph_stats(adj_mx, name):
    """算法改动: 除了 node/edge 数量, 还计算:
    1. degree distribution (mean/std/max)
    2. spectral gap 的近似 (最大特征值与第二大特征值之差)
    3. 局部聚类系数均值
    这些统计量对理解图的连通性和消息传播特性很重要"""
    binary = (adj_mx != 0).astype(np.float32)
    n = binary.shape[0]
    edge = int(binary.sum())
    degrees = binary.sum(axis=1)

    print(f"==================== {name} ====================")
    print(f"# Node: {n}")
    print(f"# Edge: {edge}")
    print(f"Degree  mean={degrees.mean():.2f}  std={degrees.std():.2f}  "
          f"max={degrees.max():.0f}  min={degrees.min():.0f}")

    # 算法改动: isolated nodes 检测
    isolated = int((degrees == 0).sum())
    if isolated > 0:
        print(f"WARNING: {isolated} isolated nodes detected!")

    # 算法改动: spectral gap 近似
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        sp_adj = csr_matrix(binary)
        if n > 2:
            eigenvalues = eigsh(sp_adj.astype(np.float64), k=min(6, n - 1),
                                return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)[::-1]
            if len(eigenvalues) >= 2:
                spectral_gap = eigenvalues[0] - eigenvalues[1]
                print(f"Spectral gap (approx): {spectral_gap:.4f}  "
                      f"top eigenvalues: {eigenvalues[:3]}")
    except Exception as e:
        if _DBG_DESC:
            print(f"[DBG-DESC] spectral computation failed: {e}")

    # 算法改动: 局部聚类系数
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(binary[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_coeffs.append(0.0)
            continue
        sub = binary[np.ix_(neighbors, neighbors)]
        triangles = sub.sum() / 2.0
        cc = 2.0 * triangles / (k * (k - 1))
        clustering_coeffs.append(cc)
    cc_arr = np.array(clustering_coeffs)
    print(f"Clustering coeff  mean={cc_arr.mean():.4f}  "
          f"std={cc_arr.std():.4f}")
    print()


# METR-LA
file_path = "datasets/sensor_graph/adj_mx_la.pkl"
adj_mx = load_pickle(file_path)[2]
_graph_stats(adj_mx, "METR-LA")

# PEMS-BAY
file_path = "datasets/sensor_graph/adj_mx_bay.pkl"
adj_mx = load_pickle(file_path)[2]
_graph_stats(adj_mx, "PEMS-BAY")

# PEMS04
file_path = "datasets/sensor_graph/adj_mx_04.pkl"
adj_mx = load_pickle(file_path)
_graph_stats(adj_mx, "PEMS04")

# PEMS08
file_path = "datasets/sensor_graph/adj_mx_08.pkl"
adj_mx = load_pickle(file_path)
_graph_stats(adj_mx, "PEMS08")
