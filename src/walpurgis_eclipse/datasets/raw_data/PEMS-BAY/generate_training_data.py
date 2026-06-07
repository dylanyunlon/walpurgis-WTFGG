"""Generate PEMS-BAY training data."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from datasets.raw_data._gen_speed_common import generate_speed_data
if __name__ == '__main__':
    base = os.path.dirname(__file__)
    generate_speed_data(os.path.join(base, 'PEMS_BAY.h5'),
                        os.path.join(base, '..', '..', 'PEMS-BAY'))
