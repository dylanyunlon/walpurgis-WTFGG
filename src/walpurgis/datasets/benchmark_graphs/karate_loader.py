"""
benchmark_graphs/karate_loader — Zachary Karate Club 图数据集加载器

迁移自 cugraph-gnn@43a80e8 (datasets/karate.csv, Alexandra Barghi, 2025-06-09)
格式: <src> <dst> <weight>，无表头，空格分隔，双向边（正向+反向均已列出）

WALPURGIS_DEBUG: 设置环境变量 WALPURGIS_DEBUG=1 开启断点式诊断打印。
"""

import os
import sys
import csv
from pathlib import Path
from typing import Tuple, List, Optional

_WDBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'

# 数据文件相对本模块的位置
_KARATE_CSV = Path(__file__).parent / 'karate.csv'


def _dbg(tag: str, msg: str) -> None:
    """WALPURGIS_DEBUG 断点打印，仅在 WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _WDBG:
        print(f'[WALPURGIS_DBG:{tag}] {msg}', file=sys.stderr)


def load_karate_edges(
    csv_path: Optional[Path] = None,
    directed: bool = False,
) -> Tuple[List[int], List[int], List[float]]:
    """
    从 karate.csv 加载边列表。

    上游 CSV 已包含正向+反向边；directed=False 时去重只保留 src<dst 的半边，
    directed=True 时返回全部原始行（含双向重复）。

    Returns
    -------
    (src_list, dst_list, weight_list) — 三个等长列表。
    """
    path = csv_path or _KARATE_CSV
    _dbg('load_karate_edges', f'path={path}  directed={directed}')

    if not path.exists():
        raise FileNotFoundError(
            f'[karate_loader] 找不到数据文件: {path}\n'
            '请确认 datasets/benchmark_graphs/karate.csv 已正确迁移。'
        )

    src_list: List[int] = []
    dst_list: List[int] = []
    weight_list: List[float] = []
    skipped = 0

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for lineno, row in enumerate(reader, start=1):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 2:
                _dbg('load_karate_edges', f'SKIP malformed row {lineno}: {row!r}')
                skipped += 1
                continue
            try:
                s, d = int(row[0]), int(row[1])
                w = float(row[2]) if len(row) >= 3 else 1.0
            except ValueError:
                _dbg('load_karate_edges', f'SKIP bad parse row {lineno}: {row!r}')
                skipped += 1
                continue

            if directed or s < d:          # 无向模式只保留半边
                src_list.append(s)
                dst_list.append(d)
                weight_list.append(w)

    _dbg('load_karate_edges',
         f'rows_kept={len(src_list)}  skipped={skipped}  '
         f'nodes_seen={len(set(src_list + dst_list))}  '
         f'weight_range=[{min(weight_list):.2f},{max(weight_list):.2f}]')

    if len(src_list) == 0:
        raise RuntimeError('[karate_loader] 解析后边数为 0，数据文件可能已损坏。')

    return src_list, dst_list, weight_list


def load_karate_adj(
    csv_path: Optional[Path] = None,
    num_nodes: int = 34,          # Zachary karate club 固定 34 节点（0-indexed 0..33）
) -> 'list[list[float]]':
    """
    构造稠密邻接矩阵（对称，float 权重）。

    Parameters
    ----------
    num_nodes : 节点数，karate club 标准为 34（节点 ID 0–33）。
                若数据实际节点数更大，自动扩展。

    Returns
    -------
    adj : List[List[float]]，shape (num_nodes, num_nodes)
    """
    _dbg('load_karate_adj', f'num_nodes={num_nodes}')

    src_list, dst_list, weight_list = load_karate_edges(csv_path, directed=False)

    # 自动扩展以防节点 ID 越界
    actual_max = max(max(src_list), max(dst_list)) + 1
    if actual_max > num_nodes:
        _dbg('load_karate_adj',
             f'WARNING: actual_max={actual_max} > num_nodes={num_nodes}，自动扩展')
        num_nodes = actual_max

    adj = [[0.0] * num_nodes for _ in range(num_nodes)]
    for s, d, w in zip(src_list, dst_list, weight_list):
        adj[s][d] = w
        adj[d][s] = w          # 对称填充

    # WALPURGIS_DEBUG: 检查对角线、密度
    _dbg('load_karate_adj',
         f'shape=({num_nodes},{num_nodes})  '
         f'nonzero={sum(1 for r in adj for v in r if v != 0)}  '
         f'diagonal_sum={sum(adj[i][i] for i in range(num_nodes))}')

    return adj


def karate_graph_info() -> dict:
    """
    返回 Zachary Karate Club 图的元信息字典，方便 walpurgis 配置层引用。

    >>> info = karate_graph_info()
    >>> info['num_nodes']
    34
    """
    src_list, dst_list, weight_list = load_karate_edges()
    nodes = set(src_list + dst_list)
    info = {
        'name': 'zachary_karate_club',
        'source_commit': '43a80e8',                # cugraph-gnn upstream
        'num_nodes': len(nodes),
        'num_edges_undirected': len(src_list),    # 去重后半边数
        'node_id_range': (min(nodes), max(nodes)),
        'all_weights_unity': all(w == 1.0 for w in weight_list),
    }
    _dbg('karate_graph_info', str(info))
    return info


# ── 快速自测 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json
    # 强制开启 DEBUG 输出
    os.environ['WALPURGIS_DEBUG'] = '1'
    # 重新读取环境变量（模块已加载，需手动翻转 _WDBG）
    import importlib, sys as _sys
    _WDBG = True   # noqa: F811

    print('=== karate_loader self-test ===')
    info = karate_graph_info()
    print(json.dumps(info, indent=2))

    srcs, dsts, ws = load_karate_edges(directed=False)
    print(f'undirected edges: {len(srcs)}')

    adj = load_karate_adj()
    row0_nonzero = [(j, adj[0][j]) for j in range(len(adj[0])) if adj[0][j] != 0]
    print(f'node-0 neighbors: {row0_nonzero}')
    print('=== PASS ===')
