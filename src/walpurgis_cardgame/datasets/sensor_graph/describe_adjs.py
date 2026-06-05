"""
describe_adjs — CardGame变体
邻接矩阵描述工具

算法改写 (vs upstream):
  1. 谱分析: 计算Laplacian特征值间隙 (algebraic connectivity)
  2. 聚类系数: 估算局部三角形密度
  3. PageRank中心性: 用幂迭代计算top-k节点
"""
import pickle
import numpy as np
import sys
import os

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _dbg(tag, tensor, module="describe_adjs"):
    if not _CG_DEBUG:
        return
    if hasattr(tensor, 'shape'):
        arr = np.asarray(tensor)
        msg = (f"[CG-DBG:{tag}] shape={list(arr.shape)} dtype={arr.dtype} "
               f"min={arr.min():.6f} max={arr.max():.6f} "
               f"mean={arr.mean():.6f} std={arr.std():.6f}")
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            msg += f" *** NaN={nan_count} ***"
        if inf_count > 0:
            msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def _pagerank(adj, alpha=0.85, max_iter=50, tol=1e-6):
    """幂迭代PageRank"""
    n = adj.shape[0]
    out_degree = adj.sum(axis=1)
    out_degree[out_degree == 0] = 1.0
    M = (adj / out_degree[:, None]).T
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = alpha * M @ pr + (1 - alpha) / n
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    return pr


def _clustering_coefficient(binary_adj):
    """估算平均聚类系数"""
    n = binary_adj.shape[0]
    coeffs = []
    for i in range(n):
        neighbors = np.where(binary_adj[i])[0]
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        sub = binary_adj[np.ix_(neighbors, neighbors)]
        triangles = sub.sum() / 2
        possible = k * (k - 1) / 2
        coeffs.append(triangles / possible)
    return np.array(coeffs)


def _spectral_gap(adj):
    """计算归一化Laplacian的代数连通性 (第二小特征值)"""
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-10)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    try:
        eigvals = np.linalg.eigvalsh(L_norm)
        eigvals = np.sort(eigvals)
        lambda2 = eigvals[1] if n > 1 else 0.0
        return float(lambda2), eigvals
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(n)


def describe_adj(adj_path):
    """描述邻接矩阵的结构特性"""
    with open(adj_path, 'rb') as f:
        try:
            sensor_ids, id_to_ind, adj_mx = pickle.load(f)
        except Exception:
            try:
                f.seek(0)
                adj_mx = pickle.load(f)
            except Exception:
                f.seek(0)
                adj_mx = pickle.load(f, encoding='latin1')
            sensor_ids = list(range(adj_mx.shape[0]))

    if not isinstance(adj_mx, np.ndarray):
        adj_mx = np.array(adj_mx)

    n = adj_mx.shape[0]
    _dbg("adj_raw", adj_mx)

    print(f"Shape: {adj_mx.shape}")
    print(f"Nodes: {n}")
    print(f"Value range: [{adj_mx.min():.4f}, {adj_mx.max():.4f}]")

    edge_count = (adj_mx != 0).sum()
    density = edge_count / (n * n)
    print(f"Edges (nonzero): {edge_count}")
    print(f"Density: {density:.4f}")
    print(f"Symmetric: {np.allclose(adj_mx, adj_mx.T)}")

    # 度分布
    binary = (adj_mx > 0.01).astype(float)
    degrees = binary.sum(axis=1)
    print(f"Degree — min={degrees.min():.0f} max={degrees.max():.0f} "
          f"mean={degrees.mean():.1f} median={np.median(degrees):.1f} "
          f"std={degrees.std():.1f}")
    isolated = int((degrees == 0).sum())
    if isolated > 0:
        print(f"  WARNING: {isolated} isolated nodes (degree=0)")

    # 连通分量 (BFS)
    visited = set()
    components = 0
    for start in range(n):
        if start in visited:
            continue
        components += 1
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            neighbors = np.where(binary[node] | binary[:, node])[0]
            queue.extend(int(nb) for nb in neighbors if nb not in visited)
    print(f"Connected components: {components}")

    # 谱分析
    if n <= 2000:
        lambda2, eigvals = _spectral_gap(adj_mx)
        print(f"Spectral gap (algebraic connectivity): {lambda2:.6f}")
        _dbg("eigvals_top5", eigvals[:5])
    else:
        print("Spectral gap: skipped (n > 2000)")

    # 聚类系数
    cc = _clustering_coefficient(binary)
    print(f"Clustering coeff — mean={cc.mean():.4f} "
          f"min={cc.min():.4f} max={cc.max():.4f}")

    # PageRank top-5
    if n > 0:
        pr = _pagerank(adj_mx)
        top_k = min(5, n)
        top_idx = np.argsort(pr)[-top_k:][::-1]
        pr_str = ", ".join(f"node{i}={pr[i]:.4f}" for i in top_idx)
        print(f"PageRank top-{top_k}: {pr_str}")
        _dbg("pagerank_all", pr)

    return adj_mx


if __name__ == "__main__":
    if len(sys.argv) > 1:
        describe_adj(sys.argv[1])
    else:
        default_paths = [
            "datasets/sensor_graph/adj_mx_la.pkl",
            "datasets/sensor_graph/adj_mx_bay.pkl",
            "datasets/sensor_graph/adj_mx_04.pkl",
            "datasets/sensor_graph/adj_mx_08.pkl",
            "datasets/sensor_graph/adj_mx_synth.pkl",
        ]
        for p in default_paths:
            if os.path.exists(p):
                print(f"\n{'='*20} {p} {'='*20}")
                describe_adj(p)
