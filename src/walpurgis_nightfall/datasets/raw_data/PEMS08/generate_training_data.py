"""PEMS08 generate_training_data — Nightfall, 调用_gen_flow_common"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets._gen_flow_common import generate_train_val_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='datasets/PEMS08')
    parser.add_argument("--traffic_df_filename", type=str, default='datasets/raw_data/PEMS08/PEMS08.npz')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args, dataset_name="PEMS08", train_ratio=0.6, test_ratio=0.2)
