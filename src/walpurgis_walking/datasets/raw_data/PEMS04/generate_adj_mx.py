import numpy as np
import csv
import pickle
import json
import os


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx
                           for idx, i in enumerate(
                               f.read().strip().split('\n'))}
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices,
                                    id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx
                           for idx, i in enumerate(
                               f.read().strip().split('\n'))}
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


# ========== 改动1: 距离矩阵 → RBF连续权重 ==========
# upstream: A是0/1二值, distanceA单独存但不参与后续GNN计算
# walpurgis改动: 将距离转成连续权重 exp(-d²/2σ²), σ=非零距离中位数
def _distance_to_rbf_weights(A, distaneA):
    nz_dists = distaneA[distaneA > 0].flatten()
    if len(nz_dists) == 0:
        return A, distaneA

    sigma = float(np.median(nz_dists))
    if sigma < 1e-10:
        sigma = 1.0

    # 在有边的位置, 用 RBF 替代二值1
    rbf_A = np.zeros_like(A)
    mask = distaneA > 0
    rbf_A[mask] = np.exp(-distaneA[mask] ** 2 / (2.0 * sigma ** 2))

    print(f"[walpurgis adj] RBF: σ={sigma:.2f}, "
          f"weight range [{rbf_A[mask].min():.4f}, {rbf_A[mask].max():.4f}]")
    return rbf_A, distaneA


# ========== 改动2: k-NN 稀疏化 ==========
# upstream: 保留CSV中所有边, 完全取决于原始数据密度
# walpurgis改动: 每节点只保留权重最大的k个邻居
_KNN_K = 15


def _knn_sparsify(adj, k=_KNN_K):
    n = adj.shape[0]
    sparse = np.zeros_like(adj)
    for i in range(n):
        row = adj[i]
        nz = np.count_nonzero(row)
        if nz <= k:
            sparse[i] = row
        else:
            topk_idx = np.argpartition(row, -k)[-k:]
            sparse[i, topk_idx] = row[topk_idx]

    before = np.count_nonzero(adj)
    after = np.count_nonzero(sparse)
    print(f"[walpurgis adj] kNN(k={k}): edges {before} → {after} "
          f"({100*after/max(before,1):.1f}%)")
    return sparse


# ========== 改动3: 自适应阈值剪枝 ==========
# upstream: 无阈值, 保留所有非零边
# walpurgis改动: 权重 < (mean - 2*std) 的弱边删掉, 减少噪声连接
def _adaptive_threshold_prune(adj):
    nz_vals = adj[adj > 0]
    if len(nz_vals) < 10:
        return adj  # 太少不做
    mu = nz_vals.mean()
    sigma = nz_vals.std()
    threshold = max(mu - 2.0 * sigma, 1e-6)

    pruned = adj.copy()
    weak_mask = (pruned > 0) & (pruned < threshold)
    n_pruned = weak_mask.sum()
    pruned[weak_mask] = 0.0

    print(f"[walpurgis adj] Adaptive prune: threshold={threshold:.4f} "
          f"(μ={mu:.4f}, σ={sigma:.4f}), pruned {n_pruned} weak edges")
    return pruned


# ========== 改动4: 度分布审计 ==========
# upstream: 只输出总 edge 数
# walpurgis改动: 输出 degree 直方图、孤立节点数、最大度、平均度
def _degree_audit(adj, name=""):
    degrees = np.count_nonzero(adj, axis=1)
    isolated = int(np.sum(degrees == 0))
    avg_deg = float(degrees.mean())
    max_deg = int(degrees.max())
    min_deg = int(degrees.min())

    # 直方图 bin
    hist_bins = [0, 1, 5, 10, 20, 50, 100, 500]
    hist_counts = []
    for lo, hi in zip(hist_bins[:-1], hist_bins[1:]):
        cnt = int(np.sum((degrees >= lo) & (degrees < hi)))
        hist_counts.append(f"[{lo},{hi}):{cnt}")

    overflow = int(np.sum(degrees >= hist_bins[-1]))
    hist_counts.append(f"[{hist_bins[-1]},∞):{overflow}")

    print(f"[walpurgis adj] {name} Degree audit: "
          f"nodes={adj.shape[0]}, edges={np.count_nonzero(adj)}, "
          f"isolated={isolated}, deg_range=[{min_deg},{max_deg}], "
          f"avg={avg_deg:.1f}")
    print(f"  Histogram: {', '.join(hist_counts)}")

    return {
        'nodes': int(adj.shape[0]),
        'edges': int(np.count_nonzero(adj)),
        'isolated': isolated,
        'min_degree': min_deg,
        'max_degree': max_deg,
        'avg_degree': round(avg_deg, 2),
    }


# ========== 改动5: 自环权重 = 行最大值 ==========
# upstream: 自环加 identity (权重=1), 不考虑边权尺度
# walpurgis改动: 自环权重设为该行的最大权重, 让自连接不被邻居淹没
def _add_weighted_self_loop(adj):
    row_max = adj.max(axis=1)
    # 没有邻居的节点自环设1
    row_max[row_max == 0] = 1.0
    np.fill_diagonal(adj, row_max)
    print(f"[walpurgis adj] Self-loop: weight=row_max, "
          f"range [{row_max.min():.4f}, {row_max.max():.4f}]")
    return adj


# ================= main =================
direction = True
distance_df_filename = "datasets/raw_data/PEMS04/PEMS04.csv"
num_of_vertices = 307
id_filename = None

if direction:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(
        distance_df_filename, num_of_vertices, id_filename=None)
else:
    adj_mx, distance_mx = get_adjacency_matrix(
        distance_df_filename, num_of_vertices, id_filename=None)

# pipeline: RBF → kNN → threshold → self-loop → audit
adj_mx, distance_mx = _distance_to_rbf_weights(adj_mx, distance_mx)
adj_mx = _knn_sparsify(adj_mx)
adj_mx = _adaptive_threshold_prune(adj_mx)

add_self_loop = True  # upstream default False; default True
if add_self_loop:
    adj_mx = _add_weighted_self_loop(adj_mx)

stats = _degree_audit(adj_mx, name="PEMS04")

os.makedirs("datasets/sensor_graph", exist_ok=True)
pickle.dump(adj_mx,
            open("datasets/sensor_graph/adj_mx_04.pkl", 'wb'))
pickle.dump(distance_mx,
            open("datasets/sensor_graph/adj_mx_04_distance.pkl", 'wb'))

# 保存审计JSON
with open("datasets/sensor_graph/adj_mx_04_audit.json", 'w') as f:
    json.dump(stats, f, indent=2)
print("[walpurgis] PEMS04 adj saved with audit.")
