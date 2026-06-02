"""Generate METR-LA train/val/test splits."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _gen_speed_common import run_speed_gen

if __name__ == "__main__":
    run_speed_gen(
        dataset_tag="METR-LA",
        output_dir="datasets/METR-LA",
        h5_path="datasets/raw_data/METR-LA/metr-la.h5",
    )
