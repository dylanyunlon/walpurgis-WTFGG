"""
log — Nightfall变体
改写: 简化日志, 去掉文件复制, 改用轻量timestamp日志
"""
import time
import os


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
        self.log_dir = 'log/'
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.session_dir = os.path.join(self.log_dir, f"{model_name}_{dataset}_{cur_time}")
        os.makedirs(self.session_dir, exist_ok=True)
        # 写入元数据
        with open(os.path.join(self.session_dir, 'meta.txt'), 'w') as f:
            f.write(f"model: {model_name}\n")
            f.write(f"dataset: {dataset}\n")
            f.write(f"start: {cur_time}\n")
            f.write(f"variant: nightfall\n")

    def _print(self, dic, note=None, ban=[]):
        print("=" * 16 + f" {note} " + "=" * 17)
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("-" * 44)

    def print_model_args(self, model_args, ban=[]):
        self._print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self._print(optim_args, note='optim args', ban=ban)
