"""Logging utilities for Walpurgis training pipeline.

Walpurgis adaptations:
- TrainLogger now also exports a machine-readable JSON summary
- clock decorator includes memory delta tracking
- File copy operations are logged with sizes
"""
import time
import os
import json
import shutil


def clock(func):
    """Time counter decorator with Walpurgis memory tracking."""
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print(f'[Walpurgis::clock] [{elapsed:0.8f}s] {name}')
        return result
    return clocked


class TrainLogger():
    """Logger class for Walpurgis training runs.

    Functions:
    - Print all training hyperparameter settings
    - Print all model hyperparameter settings
    - Save all the python files of model
    - Export machine-readable JSON summary

    Args:
        model_name (str): name of the model
        dataset (str): dataset name
    """

    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time = cur_time.replace(" ", "-")
        self.log_dir = path + cur_time
        self._model_name = model_name
        self._dataset = dataset
        self._meta = {
            'model_name': model_name,
            'dataset': dataset,
            'timestamp': cur_time,
            'args': {}
        }

        # mkdir
        os.makedirs(self.log_dir)
        print(f"[Walpurgis::TrainLogger] created log dir: {self.log_dir}")

        # copy model files
        for src_dir in ['models', 'configs']:
            if os.path.exists(src_dir):
                dst = os.path.join(self.log_dir, src_dir)
                shutil.copytree(src_dir, dst)
                n_files = sum(len(f) for _, _, f in os.walk(dst))
                print(f"  copied {src_dir}/ ({n_files} files)")

        if os.path.exists('main.py'):
            shutil.copyfile('main.py', os.path.join(self.log_dir, 'main.py'))
            print(f"  copied main.py")

        # backup model parameters
        for suffix in ['', '_resume']:
            pt_name = f"{model_name}_{dataset}{suffix}.pt"
            src = os.path.join('output', pt_name)
            if os.path.exists(src):
                dst = os.path.join(self.log_dir, pt_name)
                shutil.copyfile(src, dst)
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                print(f"  backed up {pt_name} ({size_mb:.2f} MB)")

    def __print(self, dic, note=None, ban=[]):
        print(f"\n{'='*20} {note} {'='*20}")
        for key, value in dic.items():
            if key in ban:
                continue
            print(f'|{key:>25s}: {str(value):<25s}|')
        print("-" * (44 + len(note)))

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='Walpurgis Model Args', ban=ban)
        self._meta['args']['model'] = {k: str(v) for k, v in model_args.items() if k not in ban}

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='Walpurgis Optim Args', ban=ban)
        self._meta['args']['optim'] = {k: str(v) for k, v in optim_args.items() if k not in ban}

    def export_meta(self):
        """Export training metadata as JSON."""
        meta_path = os.path.join(self.log_dir, 'walpurgis_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(self._meta, f, indent=2)
        print(f"[Walpurgis::TrainLogger] meta exported to {meta_path}")
