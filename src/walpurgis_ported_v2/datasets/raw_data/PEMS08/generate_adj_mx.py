"""Build PEMS08 adjacency matrix from raw CSV."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.raw_data._gen_adj_common import build_adj, save_adj

if __name__ == "__main__":
    adj, dist = build_adj(
        csv_path="datasets/raw_data/PEMS08/PEMS08.csv",
        n_nodes=170,
        bidirectional=True,
    )
    save_adj(adj, dist,
             "datasets/sensor_graph/adj_mx_08.pkl",
             "datasets/sensor_graph/adj_mx_08_distance.pkl")
