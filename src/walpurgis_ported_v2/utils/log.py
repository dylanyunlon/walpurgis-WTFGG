"""
Logging and experiment snapshot utilities.
"""

import time
import os
import shutil
import sys

_DBG_LOG = ("--debug-log" in sys.argv) or False


def clock(func):
    """Decorator that prints wall-clock execution time of *func*."""
    def timed_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        fn_name = func.__name__
        print(f'[{elapsed:0.8f}s] {fn_name}')
        return result
    return timed_wrapper


class TrainLogger:
    """
    Snapshot current code and config alongside the experiment for
    post-mortem debugging.  Also pretty-prints hyperparameter tables.
    """

    def __init__(self, model_name, dataset_name, log_root='log/'):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self._log_dir = os.path.join(log_root, timestamp)
        os.makedirs(self._log_dir, exist_ok=True)

        # snapshot source tree
        for folder in ('models', 'configs'):
            src = folder
            dst = os.path.join(self._log_dir, folder)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
        if os.path.isfile('main.py'):
            shutil.copyfile('main.py', os.path.join(self._log_dir, 'main.py'))

        # try to copy best & resume checkpoints
        for suffix in ('', '_resume'):
            ckpt_name = f"{model_name}_{dataset_name}{suffix}.pt"
            ckpt_src = os.path.join('output', ckpt_name)
            if os.path.isfile(ckpt_src):
                shutil.copyfile(ckpt_src, os.path.join(self._log_dir, ckpt_name))

        if _DBG_LOG:
            print(f"[DBG:log] TrainLogger created  dir={self._log_dir}")

    # ──── pretty-printers ────

    def _print_table(self, mapping, title, skip_keys=()):
        border = "=" * 20
        print(f"\n{border} {title} {border}")
        for k, v in mapping.items():
            if k in skip_keys:
                continue
            print(f'| {k:>24s} : {str(v):<24s} |')
        print("-" * (len(border) * 2 + len(title) + 2))

    def print_model_args(self, model_args, ban=()):
        self._print_table(model_args, 'Model Hyperparameters', skip_keys=ban)

    def print_optim_args(self, optim_args, ban=()):
        self._print_table(optim_args, 'Optimizer Settings', skip_keys=ban)
