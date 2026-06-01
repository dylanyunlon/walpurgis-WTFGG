"""
Walpurgis v2 Training Logger — Structured JSON + Run Archival
===============================================================
Delta: adds `.log_epoch(epoch, metrics)` for machine-readable JSONL
output alongside human-readable console logs.
"""
import time
import os
import json
import shutil
import functools


def clock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[clock] {func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


class TrainLogger:
    """Training run logger with JSON epoch logging.

    New: .log_epoch(epoch, metrics_dict) appends a JSONL line to
    `{log_dir}/epochs.jsonl` for automated downstream analysis.
    """

    def __init__(self, model_name, dataset):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.log_dir = f"log/{model_name}_{dataset}_{ts}"
        self._model = model_name
        self._dataset = dataset
        self._meta = {"model": model_name, "dataset": dataset, "timestamp": ts, "args": {}}

        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[TrainLogger] log dir: {self.log_dir}")

        for src_dir in ["models", "configs"]:
            if os.path.exists(src_dir):
                dst = os.path.join(self.log_dir, src_dir)
                shutil.copytree(src_dir, dst)

        if os.path.exists("main.py"):
            shutil.copyfile("main.py", os.path.join(self.log_dir, "main.py"))

        for suffix in ["", "_resume"]:
            pt = f"{model_name}_{dataset}{suffix}.pt"
            src = os.path.join("output", pt)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(self.log_dir, pt))

        self._jsonl_path = os.path.join(self.log_dir, "epochs.jsonl")

    def _format_table(self, params, title, exclude=None):
        exclude = exclude or []
        print(f"\n{'═'*20} {title} {'═'*20}")
        for key, value in params.items():
            if key not in exclude:
                print(f"  {key:>25s}: {str(value):<30s}")
        print(f"{'─'*50}")

    def print_model_args(self, model_args, ban=None):
        ban = ban or []
        self._format_table(model_args, "Model Args", exclude=ban)
        self._meta["args"]["model"] = {k: str(v) for k, v in model_args.items() if k not in ban}

    def print_optim_args(self, optim_args, ban=None):
        ban = ban or []
        self._format_table(optim_args, "Optim Args", exclude=ban)
        self._meta["args"]["optim"] = {k: str(v) for k, v in optim_args.items() if k not in ban}

    def log_epoch(self, epoch, metrics):
        """Append JSONL line for machine-readable epoch tracking."""
        entry = {"epoch": epoch, "ts": time.time(), **metrics}
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def export_meta(self):
        path = os.path.join(self.log_dir, "meta.json")
        with open(path, "w") as f:
            json.dump(self._meta, f, indent=2)
        print(f"[TrainLogger] meta → {path}")
