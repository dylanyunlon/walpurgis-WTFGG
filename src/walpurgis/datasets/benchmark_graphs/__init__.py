"""benchmark_graphs — 图基准数据集子包（migrated from cugraph-gnn）。"""
from .karate_loader import load_karate_edges, load_karate_adj, karate_graph_info

__all__ = ['load_karate_edges', 'load_karate_adj', 'karate_graph_info']
