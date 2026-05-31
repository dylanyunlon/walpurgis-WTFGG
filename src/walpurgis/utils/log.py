"""
Walpurgis Training Logger — Run Tracking and File Archival
============================================================
Derived from D2STGNN log.py with ~20% restructuring.

Changes:
  1. Refactored _print → _format_table for reusable formatting
  2. export_meta() writes JSON summary for automated analysis
  3. clock() decorator uses functools.wraps for proper introspection
  4. Log dir includes dataset name for easier identification
"""
import time
import os
import json
import shutil
import functools


def clock(func):
    """Timing decorator — preserves function metadata via functools.wraps."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f'[clock] {func.__name__}: {elapsed:.6f}s')
        return result
    return wrapper


class TrainLogger:
    """Training run logger with file archival and JSON export.
    
    Creates a timestamped log directory, copies model source and configs,
    and tracks hyperparameters for reproducibility.
    
    Debug usage:
        logger.export_meta()  # write JSON summary anytime
    """

    def __init__(self, model_name, dataset):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.log_dir = f'log/{model_name}_{dataset}_{ts}'
        self._model = model_name
        self._dataset = dataset
        self._meta = {
            'model': model_name,
            'dataset': dataset,
            'timestamp': ts,
            'args': {}
        }

        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[TrainLogger] log dir: {self.log_dir}")

        # Archive source files
        for src_dir in ['models', 'configs']:
            if os.path.exists(src_dir):
                dst = os.path.join(self.log_dir, src_dir)
                shutil.copytree(src_dir, dst)
                n_files = sum(len(files) for _, _, files in os.walk(dst))
                print(f"  archived {src_dir}/ ({n_files} files)")

        if os.path.exists('main.py'):
            shutil.copyfile('main.py', os.path.join(self.log_dir, 'main.py'))
            print(f"  archived main.py")

        # Backup checkpoints
        for suffix in ['', '_resume']:
            pt_name = f"{model_name}_{dataset}{suffix}.pt"
            src = os.path.join('output', pt_name)
            if os.path.exists(src):
                dst = os.path.join(self.log_dir, pt_name)
                shutil.copyfile(src, dst)
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                print(f"  backed up {pt_name} ({size_mb:.2f}MB)")

    def _format_table(self, params, title, exclude=None):
        """Print parameter table with aligned columns."""
        exclude = exclude or []
        print(f"\n{'═'*20} {title} {'═'*20}")
        for key, value in params.items():
            if key not in exclude:
                print(f"  {key:>25s}: {str(value):<30s}")
        print(f"{'─'*50}")

    def print_model_args(self, model_args, ban=None):
        ban = ban or []
        self._format_table(model_args, 'Model Args', exclude=ban)
        self._meta['args']['model'] = {
            k: str(v) for k, v in model_args.items() if k not in ban
        }

    def print_optim_args(self, optim_args, ban=None):
        ban = ban or []
        self._format_table(optim_args, 'Optim Args', exclude=ban)
        self._meta['args']['optim'] = {
            k: str(v) for k, v in optim_args.items() if k not in ban
        }

    def export_meta(self):
        """Write training metadata to JSON — call anytime."""
        path = os.path.join(self.log_dir, 'meta.json')
        with open(path, 'w') as f:
            json.dump(self._meta, f, indent=2)
        print(f"[TrainLogger] meta → {path}")
