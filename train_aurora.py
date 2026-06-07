import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from walpurgis_aurora.main import main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Aurora D2STGNN variant')
    parser.add_argument('--config', type=str, default='configs/SYNTH.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'walpurgis_aurora', args.config)
    main(config_path, args.device, args.epochs)
