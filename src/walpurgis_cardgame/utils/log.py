"""
D2STGNN CardGame variant — log.py
Algorithm changes vs upstream:
  1. JSONL structured logging: each event written as a single JSON line to a .jsonl file
  2. Per-epoch metric CSV dump: training/validation metrics appended to metrics.csv
"""

import os
import sys
import time
import json
import csv
import shutil

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time = cur_time.replace(" ", "-")
        self._log_dir = os.path.join(path, cur_time)
        os.makedirs(self._log_dir, exist_ok=True)

        # --- CARDGAME: JSONL structured log ---
        self._jsonl_path = os.path.join(self._log_dir, 'events.jsonl')
        self._log_event('init', {'model_name': model_name, 'dataset': dataset,
                                  'timestamp': cur_time})

        # --- CARDGAME: per-epoch metric CSV ---
        self._csv_path = os.path.join(self._log_dir, 'metrics.csv')
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_mae', 'train_mape', 'train_rmse',
                             'val_mae', 'val_mape', 'val_rmse', 'lr', 'timestamp'])
        _dbg("log.csv_init", self._csv_path, "log")

        # copy source for reproducibility
        try:
            if os.path.exists('models'):
                shutil.copytree('models', os.path.join(self._log_dir, "models"), dirs_exist_ok=True)
            if os.path.exists('configs'):
                shutil.copytree('configs', os.path.join(self._log_dir, "configs"), dirs_exist_ok=True)
            if os.path.exists('main.py'):
                shutil.copyfile('main.py', os.path.join(self._log_dir, "main.py"))
        except Exception:
            pass

        # backup model parameters
        try:
            shutil.copyfile('output/' + model_name + "_" + dataset + ".pt",
                            os.path.join(self._log_dir, model_name + "_" + dataset + ".pt"))
            shutil.copyfile('output/' + model_name + "_" + dataset + "_resume.pt",
                            os.path.join(self._log_dir, model_name + "_" + dataset + "_resume.pt"))
        except Exception:
            pass

    def _log_event(self, event_type, data):
        """Write a single JSONL event line."""
        record = {'type': event_type, 'time': time.time(), **data}
        try:
            with open(self._jsonl_path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception:
            pass

    def log_epoch(self, epoch, train_mae, train_mape, train_rmse,
                  val_mae, val_mape, val_rmse, lr):
        """Log one epoch's metrics to both JSONL and CSV."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # JSONL
        self._log_event('epoch', {
            'epoch': epoch, 'train_mae': train_mae, 'train_mape': train_mape,
            'train_rmse': train_rmse, 'val_mae': val_mae, 'val_mape': val_mape,
            'val_rmse': val_rmse, 'lr': lr})
        # CSV
        try:
            with open(self._csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f'{train_mae:.6f}', f'{train_mape:.6f}',
                                 f'{train_rmse:.6f}', f'{val_mae:.6f}',
                                 f'{val_mape:.6f}', f'{val_rmse:.6f}',
                                 f'{lr:.8f}', ts])
        except Exception:
            pass
        _dbg("log.epoch", f"epoch={epoch} train_mae={train_mae:.4f} val_mae={val_mae:.4f}", "log")

    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)
        self._log_event('model_args', {k: v for k, v in model_args.items() if k not in ban})

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
        self._log_event('optim_args', {k: v for k, v in optim_args.items() if k not in ban})
