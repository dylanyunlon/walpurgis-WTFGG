"""
Training logger: snapshot source files and print hyper-params.
Ported with lightweight debug instrumentation.
"""
import time
import os
import shutil
import sys

_DBG = ("--debug-log" in sys.argv)


def clock(fn):
    """Decorator that prints wall-clock execution time of *fn*."""
    def _timed(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f'[{dt:0.8f}s] {fn.__name__}')
        return result
    return _timed


class TrainLogger:
    """Archive source + configs at the start of each run."""

    def __init__(self, model_name, dataset):
        base = 'log/'
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        run_dir = os.path.join(base, stamp)
        os.makedirs(run_dir, exist_ok=True)

        # snapshot source tree
        for subdir in ('models', 'configs'):
            dst = os.path.join(run_dir, subdir)
            if os.path.isdir(subdir):
                shutil.copytree(subdir, dst, dirs_exist_ok=True)
        if os.path.isfile('main.py'):
            shutil.copyfile('main.py', os.path.join(run_dir, 'main.py'))

        # try to back up existing checkpoints
        for suffix in ('', '_resume'):
            ckpt = f'output/{model_name}_{dataset}{suffix}.pt'
            if os.path.isfile(ckpt):
                shutil.copyfile(ckpt, os.path.join(run_dir, os.path.basename(ckpt)))

        if _DBG:
            print(f"[DBG:log] TrainLogger  run_dir={run_dir}")

    # ---- pretty-print dicts ----

    @staticmethod
    def _show_dict(d, title, skip_keys=()):
        print(f"{'='*16} {title} {'='*16}")
        for k, v in d.items():
            if k in skip_keys:
                continue
            print(f'|{k:>24s}: {str(v):>24s}|')
        print('-' * (34 + len(title)))

    def print_model_args(self, cfg, ban=()):
        self._show_dict(cfg, 'model args', skip_keys=ban)

    def print_optim_args(self, cfg, ban=()):
        self._show_dict(cfg, 'optim args', skip_keys=ban)
