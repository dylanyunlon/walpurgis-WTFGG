"""Generate METR-LA training data."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.raw_data._gen_speed_common import make_parser, split_and_save

if __name__ == "__main__":
    parser = make_parser(
        dataset="METR-LA",
        output_dir="datasets/METR-LA",
        h5_path="datasets/raw_data/METR-LA/metr-la.h5",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    split_and_save(args)
