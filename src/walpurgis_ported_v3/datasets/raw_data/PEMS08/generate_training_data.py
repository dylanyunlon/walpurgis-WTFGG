"""Generate PEMS08 train/val/test splits with MinMax normalization."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _gen_flow_common import run_flow_gen

if __name__ == "__main__":
    run_flow_gen(
        dataset_tag="PEMS08",
        output_dir="datasets/PEMS08",
        npz_path="datasets/raw_data/PEMS08/PEMS08.npz",
        train_ratio=0.6,
    )
