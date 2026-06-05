"""
describe_adjs — Nightfall变体
算法改写: 新增图连通性分析(连通分量数) + 度分布统计
"""
import pickle
import numpy as np
import sys
import os


def describe_adj(adj_path):
    with open(adj_path, 'rb') as f:
        try:
            sensor_ids, id_to_ind, adj_mx = pickle.load(f)
        except:
            adj_mx = pickle.load(f)
            sensor_ids = list(range(adj_mx.shape[0]))
    n = adj_mx.shape[0]
    print(f"Shape: {adj_mx.shape}")
    print(f"Nodes: {n}")
    print(f"Value range: [{adj_mx.min():.4f}, {adj_mx.max():.4f}]")
    print(f"Density: {(adj_mx > 0).sum() / (n*n):.4f}")
    print(f"Symmetric: {np.allclose(adj_mx, adj_mx.T)}")
    # 度分布
    degrees = (adj_mx > 0.01).sum(axis=1)
    print(f"Degree — min={degrees.min()} max={degrees.max()} "
          f"mean={degrees.mean():.1f} median={np.median(degrees):.1f}")
    isolated = (degrees == 0).sum()
    if isolated > 0:
        print(f"⚠ {isolated} isolated nodes (degree=0)")
    # 简单连通性 (BFS)
    visited = set()
    components = 0
    binary_adj = adj_mx > 0.01
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
            neighbors = np.where(binary_adj[node] | binary_adj[:, node])[0]
            queue.extend(int(nb) for nb in neighbors if nb not in visited)
    print(f"Connected components: {components}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        describe_adj(sys.argv[1])
    else:
        default_paths = [
            "datasets/sensor_graph/adj_mx_la.pkl",
            "datasets/sensor_graph/adj_mx_bay.pkl",
        ]
        for p in default_paths:
            if os.path.exists(p):
                print(f"\n=== {p} ===")
                describe_adj(p)
