"""Generate PEMS04 adjacency matrix."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from datasets.raw_data._gen_adj_common import generate_adj_from_distance
if __name__ == '__main__':
    base = os.path.dirname(__file__)
    generate_adj_from_distance(os.path.join(base, 'distance.csv'),
        os.path.join(base, '..', '..', 'sensor_graph', 'adj_mx_pems04.pkl'), 307)
