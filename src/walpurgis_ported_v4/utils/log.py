"""
Logging utilities — walpurgis_ported_v4
Modifications from upstream:
  - clock(): additionally reports peak RSS memory delta
  - TrainLogger.__print(): prefixes each block with wall-clock timestamp
  - TrainLogger.__init__(): prints total bytes copied for audit trail
"""
import time
import os
import shutil
import sys
import resource


def clock(func):
    """Time + memory profiling decorator (v4: adds RSS delta)."""
    def clocked(*args, **kw):
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_delta_kb = rss_after - rss_before
        name = func.__name__
        # v4: extended timing log
        print(f'[v4-CLOCK] [{elapsed:0.8f}s] {name}  (RSS delta: {rss_delta_kb} KB)',
              file=sys.stderr)
        return result
    return clocked


class TrainLogger:
    """Training logger — copies source snapshot and logs hyperparameters.
    v4: reports total snapshot size; timestamp on every print block.
    """

    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "-")
        log_dir = os.path.join(path, cur_time)
        os.makedirs(log_dir, exist_ok=True)

        total_bytes = 0
        # copy source trees
        for src_dir in ['models', 'configs']:
            if os.path.isdir(src_dir):
                dst = os.path.join(log_dir, src_dir)
                shutil.copytree(src_dir, dst, dirs_exist_ok=True)
                for root, _, files in os.walk(dst):
                    for f in files:
                        total_bytes += os.path.getsize(os.path.join(root, f))

        if os.path.isfile('main.py'):
            shutil.copyfile('main.py', os.path.join(log_dir, 'main.py'))
            total_bytes += os.path.getsize(os.path.join(log_dir, 'main.py'))

        # backup model checkpoints
        for suffix in ['', '_resume']:
            ckpt = f'output/{model_name}_{dataset}{suffix}.pt'
            if os.path.exists(ckpt):
                shutil.copyfile(ckpt, os.path.join(log_dir, os.path.basename(ckpt)))
                total_bytes += os.path.getsize(ckpt)

        print(f"[v4-DBG][TrainLogger] Snapshot saved to {log_dir}  "
              f"({total_bytes / 1024:.1f} KB total)", file=sys.stderr)

    def __print(self, dic, note=None, ban=[]):
        ts = time.strftime("%H:%M:%S")
        print(f"========= [{ts}] {note} =========")
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
