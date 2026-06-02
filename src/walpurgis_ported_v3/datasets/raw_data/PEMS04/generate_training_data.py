"""Generate PEMS04 train/val/test splits with MinMax normalization."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _gen_flow_common import run_flow_gen

if __name__ == "__main__":
    run_flow_gen(
        dataset_tag="PEMS04",
        output_dir="datasets/PEMS04",
        npz_path="datasets/raw_data/PEMS04/PEMS04.npz",
        train_ratio=0.6,
    )
