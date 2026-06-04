import pickle
import numpy as np
from collections import deque


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


# ========== 改动1: 图密度 ==========
# upstream: 只数 edge 总数
# walpurgis改动: edges / (n * (n-1)) 判断稀疏/稠密
def _compute_density(adj):
    n = adj.shape[0]
    edges = np.count_nonzero(adj)
    # 排除对角线
    diag_nz = np.count_nonzero(np.diag(adj))
    off_diag_edges = edges - diag_nz
    max_possible = n * (n - 1)
    density = off_diag_edges / max_possible if max_possible > 0 else 0
    return density, off_diag_edges


# ========== 改动2: 权重分布统计 ==========
# upstream: 无 — 只计二值edge
# walpurgis改动: 对非零权重算 mean/std/median/p10/p90
def _weight_stats(adj):
    nz = adj[adj > 0].flatten()
    if len(nz) == 0:
        return {}
    return {
        'count': int(len(nz)),
        'mean': float(np.mean(nz)),
        'std': float(np.std(nz)),
        'min': float(np.min(nz)),
        'p10': float(np.percentile(nz, 10)),
        'median': float(np.median(nz)),
        'p90': float(np.percentile(nz, 90)),
        'max': float(np.max(nz)),
    }


# ========== 改动3: BFS 连通分量分析 ==========
# upstream: 无连通性检查
# walpurgis改动: BFS 找所有连通分量, 输出是否全连通 + 最大分量大小
def _connected_components(adj):
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue
        # BFS
        queue = deque([start])
        visited[start] = True
        comp = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            # 双向: 出边和入边都算连通
            neighbors = set(np.where(adj[node] > 0)[0].tolist()
                            + np.where(adj[:, node] > 0)[0].tolist())
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    components.sort(key=len, reverse=True)
    return components


# ========== 改动4: 度分布 5 数概要 ==========
# upstream: 无度分析
# walpurgis改动: 出度 + 入度的 min/Q1/median/Q3/max
def _degree_summary(adj):
    out_deg = np.count_nonzero(adj, axis=1)
    in_deg = np.count_nonzero(adj, axis=0)

    def five_num(arr):
        return {
            'min': int(np.min(arr)),
            'Q1': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'Q3': float(np.percentile(arr, 75)),
            'max': int(np.max(arr)),
            'isolated': int(np.sum(arr == 0)),
        }
    return {'out_degree': five_num(out_deg),
            'in_degree': five_num(in_deg)}


# ========== 改动5: 非对称性检测 ==========
# upstream: 不检查对称性
# walpurgis改动: ||A - A^T||_F / ||A||_F, 0=完美对称, >0 有向
def _asymmetry_score(adj):
    diff = adj - adj.T
    norm_diff = np.linalg.norm(diff, 'fro')
    norm_A = np.linalg.norm(adj, 'fro')
    score = norm_diff / norm_A if norm_A > 0 else 0.0
    return score


def describe_adj(file_path, name, is_tuple=False):
    data = load_pickle(file_path)
    if is_tuple:
        adj_mx = data[2]
    else:
        adj_mx = data

    n = adj_mx.shape[0]
    density, off_edges = _compute_density(adj_mx)
    w_stats = _weight_stats(adj_mx)
    comps = _connected_components(adj_mx)
    deg = _degree_summary(adj_mx)
    asym = _asymmetry_score(adj_mx)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {name}")
    print(f"{sep}")
    print(f"  Nodes:             {n}")
    print(f"  Edges (off-diag):  {off_edges}")
    print(f"  Density:           {density:.6f} "
          f"({'sparse' if density < 0.05 else 'moderate' if density < 0.3 else 'dense'})")
    print(f"  Asymmetry score:   {asym:.6f} "
          f"({'symmetric' if asym < 0.01 else 'directed'})")

    if w_stats:
        print(f"  Weight stats:")
        print(f"    mean={w_stats['mean']:.4f}, "
              f"std={w_stats['std']:.4f}, "
              f"median={w_stats['median']:.4f}")
        print(f"    range=[{w_stats['min']:.4f}, {w_stats['max']:.4f}], "
              f"p10={w_stats['p10']:.4f}, p90={w_stats['p90']:.4f}")

    print(f"  Connected components: {len(comps)}")
    if len(comps) > 1:
        sizes = [len(c) for c in comps[:5]]
        print(f"    Top sizes: {sizes}")
    else:
        print(f"    Fully connected (1 component)")

    print(f"  Out-degree: min={deg['out_degree']['min']}, "
          f"Q1={deg['out_degree']['Q1']:.0f}, "
          f"med={deg['out_degree']['median']:.0f}, "
          f"Q3={deg['out_degree']['Q3']:.0f}, "
          f"max={deg['out_degree']['max']}, "
          f"isolated={deg['out_degree']['isolated']}")
    print(f"  In-degree:  min={deg['in_degree']['min']}, "
          f"Q1={deg['in_degree']['Q1']:.0f}, "
          f"med={deg['in_degree']['median']:.0f}, "
          f"Q3={deg['in_degree']['Q3']:.0f}, "
          f"max={deg['in_degree']['max']}, "
          f"isolated={deg['in_degree']['isolated']}")
    print(f"{sep}\n")


# ========== 逐个数据集分析 ==========
datasets = [
    ("datasets/sensor_graph/adj_mx_la.pkl", "METR-LA", True),
    ("datasets/sensor_graph/adj_mx_bay.pkl", "PEMS-BAY", True),
    ("datasets/sensor_graph/adj_mx_04.pkl", "PEMS04", False),
    ("datasets/sensor_graph/adj_mx_08.pkl", "PEMS08", False),
]

for fpath, dname, is_tup in datasets:
    try:
        describe_adj(fpath, dname, is_tuple=is_tup)
    except FileNotFoundError:
        print(f"[SKIP] {dname}: {fpath} not found")
    except Exception as e:
        print(f"[ERROR] {dname}: {e}")
