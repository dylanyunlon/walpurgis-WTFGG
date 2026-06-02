"""Build PEMS04 adjacency matrix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _gen_adj_common import build_and_save

if __name__ == "__main__":
    build_and_save(
        csv_path="datasets/raw_data/PEMS04/PEMS04.csv",
        n_nodes=307,
        adj_out="datasets/sensor_graph/adj_mx_04.pkl",
        dist_out="datasets/sensor_graph/adj_mx_04_distance.pkl",
        bidirectional=True,
        self_loop=False,
    )
