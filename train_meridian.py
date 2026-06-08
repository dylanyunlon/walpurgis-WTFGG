#!/usr/bin/env python3
"""Top-level training script for walpurgis_meridian D2STGNN variant.
Usage:
  python train_meridian.py [--dataset SYNTH|METR-LA|PEMS-BAY|PEMS04|PEMS08]
"""
import sys
import os

# ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src', 'walpurgis_meridian'))

from walpurgis_meridian.main import main

if __name__ == '__main__':
    main()
