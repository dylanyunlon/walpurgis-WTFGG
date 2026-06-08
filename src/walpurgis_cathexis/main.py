#!/usr/bin/env python
import argparse, time, torch, pickle, yaml, setproctitle
from utils.train import *; from utils.load_data import *; from utils.log import TrainLogger
from models.losses import *; from models import trainer; from models.model import D2STGNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SYNTH')
    args = parser.parse_args()
    set_config(0)
    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f: config = yaml.load(f, Loader=yaml.FullLoader)
    # ... standard D2STGNN main loop, see train_cathexis.py for actual entry
    print("Use train_cathexis.py from repo root instead.")

if __name__ == '__main__': main()
