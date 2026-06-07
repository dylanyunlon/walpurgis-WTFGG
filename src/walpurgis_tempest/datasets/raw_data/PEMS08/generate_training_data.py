"""Generate PEMS08 training data."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from datasets.raw_data._gen_flow_common import generate_flow_data
if __name__ == '__main__':
    base = os.path.dirname(__file__)
    generate_flow_data(os.path.join(base, 'PEMS08.npz'),
                       os.path.join(base, '..', '..', 'PEMS08'))
