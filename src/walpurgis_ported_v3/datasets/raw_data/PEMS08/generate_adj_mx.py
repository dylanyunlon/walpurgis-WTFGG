"""Build PEMS08 adjacency matrix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _gen_adj_common import build_and_save

if __name__ == "__main__":
    build_and_save(
        csv_path="datasets/raw_data/PEMS08/PEMS08.csv",
        n_nodes=170,
        adj_out="datasets/sensor_graph/adj_mx_08.pkl",
        dist_out="datasets/sensor_graph/adj_mx_08_distance.pkl",
        bidirectional=True,
        self_loop=False,
    )
