import time
import os
import shutil
import json
import sys

_DBG_LOG = ("--dbg-log" in sys.argv)


def clock(func):
    """算法改动: 除了计时, 还追踪 peak GPU memory (如果 CUDA 可用)"""
    def clocked(*args, **kw):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if has_cuda:
            import torch
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__

        mem_str = ""
        if has_cuda:
            import torch
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mem_str = f"  peak_gpu={peak_mb:.1f}MB"

        print(f'[{elapsed:0.8f}s]{mem_str} {name}')
        return result
    return clocked


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time = cur_time.replace(" ", "-")
        self.log_dir = path + cur_time
        os.makedirs(self.log_dir)

        shutil.copytree('models', self.log_dir + "/models")
        shutil.copytree('configs', self.log_dir + "/configs")
        shutil.copyfile('main.py', self.log_dir + "/main.py")

        try:
            shutil.copyfile(
                'output/' + model_name + "_" + dataset + ".pt",
                self.log_dir + "/" + model_name + "_" + dataset + ".pt")
            shutil.copyfile(
                'output/' + model_name + "_" + dataset + "_resume.pt",
                self.log_dir + "/" + model_name + "_" + dataset + "_resume.pt")
        except:
            pass

        # 算法改动: 用 JSON 格式保存超参数快照, 方便后续解析
        self._json_log = {}

    def _save_json(self):
        json_path = os.path.join(self.log_dir, "hparams.json")
        with open(json_path, 'w') as f:
            json.dump(self._json_log, f, indent=2, default=str)

    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        serializable = {}
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)
        print("--------------------------------------------")
        self._json_log[note] = serializable
        self._save_json()

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)
        if _DBG_LOG:
            total_keys = len(model_args)
            banned = len(ban)
            print(f"[DBG-LOG] model_args: {total_keys} keys, "
                  f"{banned} banned, saved to hparams.json")

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
        if _DBG_LOG:
            print(f"[DBG-LOG] optim_args saved to hparams.json")
