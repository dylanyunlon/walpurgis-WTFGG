import time
import os
import json
import csv
import shutil
import subprocess
from walpurgis import _dbg

_TAG = "log"


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print(f'[{elapsed:0.8f}s] {name}')
        return result
    return clocked


def _get_git_short_hash():
    """改动3: 获取当前 git short hash 用于日志目录名."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return 'nogit'


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 改动3: 目录名加 git hash
        git_hash = _get_git_short_hash()
        dir_name = f"{cur_time}_{git_hash}"
        self.log_dir = os.path.join(path, dir_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # 复制源码
        for src_dir in ['models', 'configs']:
            if os.path.exists(src_dir):
                shutil.copytree(src_dir,
                                os.path.join(self.log_dir, src_dir),
                                dirs_exist_ok=True)
        if os.path.exists('main.py'):
            shutil.copyfile('main.py',
                            os.path.join(self.log_dir, 'main.py'))

        # 备份模型权重
        for suffix in ['', '_resume']:
            pt = f'output/{model_name}_{dataset}{suffix}.pt'
            if os.path.exists(pt):
                shutil.copyfile(pt, os.path.join(self.log_dir,
                                                  os.path.basename(pt)))

        # 改动1: JSON Lines 日志文件
        self._jsonl_path = os.path.join(self.log_dir, 'train_log.jsonl')

        # 改动2: CSV metric dump
        self._csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._csv_initialized = False

        _dbg(_TAG, f"logger_init dir={self.log_dir}")
        print(f"[walpurgis TrainLogger] Log dir: {self.log_dir}")

    def log_jsonl(self, record: dict):
        """改动1: 追加一条 JSON Lines 记录."""
        record['_ts'] = time.time()
        record['_iso'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def log_epoch_metrics(self, epoch, train_loss, train_mape, train_rmse,
                          val_loss, val_mape, val_rmse, lr):
        """改动2: 每 epoch 写 CSV + JSONL."""
        row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'train_mape': f'{train_mape:.6f}',
            'train_rmse': f'{train_rmse:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'val_mape': f'{val_mape:.6f}',
            'val_rmse': f'{val_rmse:.6f}',
            'lr': f'{lr:.8f}',
        }
        # CSV
        if not self._csv_initialized:
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            self._csv_initialized = True
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        # JSONL
        self.log_jsonl({**row, '_type': 'epoch_metric'})

    def _format_table(self, dic, note, ban):
        """改动4: 结构化表格输出 — upstream 逐行 print."""
        lines = [f"╔{'═' * 44}╗",
                 f"║  {note:^40s}  ║",
                 f"╠{'═' * 22}╦{'═' * 21}╣"]
        for key, value in dic.items():
            if key in ban:
                continue
            k_str = str(key)[:20]
            v_str = str(value)[:19]
            lines.append(f"║ {k_str:<20s} ║ {v_str:<19s} ║")
        lines.append(f"╚{'═' * 22}╩{'═' * 21}╝")
        return '\n'.join(lines)

    def print_model_args(self, model_args, ban=[]):
        table = self._format_table(model_args, 'Model Args', ban)
        print(table)
        self.log_jsonl({'_type': 'model_args',
                        'args': {k: str(v) for k, v in model_args.items()
                                 if k not in ban}})

    def print_optim_args(self, optim_args, ban=[]):
        table = self._format_table(optim_args, 'Optim Args', ban)
        print(table)
        self.log_jsonl({'_type': 'optim_args',
                        'args': {k: str(v) for k, v in optim_args.items()
                                 if k not in ban}})
