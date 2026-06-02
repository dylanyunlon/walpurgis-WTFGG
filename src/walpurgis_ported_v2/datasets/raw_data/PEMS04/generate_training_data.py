"""Generate PEMS04 training data."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.raw_data._gen_flow_common import make_flow_parser, split_and_save_flow

if __name__ == "__main__":
    parser = make_flow_parser(
        dataset="PEMS04",
        output_dir="datasets/PEMS04",
        npz_path="datasets/raw_data/PEMS04/PEMS04.npz",
    )
    args = parser.parse_args()
    split_and_save_flow(args)
