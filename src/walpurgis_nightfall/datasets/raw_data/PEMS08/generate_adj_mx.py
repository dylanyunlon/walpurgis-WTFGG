"""PEMS08 generate_adj_mx — Nightfall, 调用_gen_adj_common"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from datasets._gen_adj_common import build_and_save

build_and_save(
    csv_path="datasets/raw_data/PEMS08/PEMS08.csv",
    num_vertices=170,
    adj_out="datasets/sensor_graph/adj_mx_08.pkl",
    dist_out="datasets/sensor_graph/adj_mx_08_distance.pkl",
    direction=True, add_self_loop=False)
