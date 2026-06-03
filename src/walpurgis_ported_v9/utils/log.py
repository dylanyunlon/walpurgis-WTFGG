"""
log.py — v9 port
Algo delta:
  1. 训练日志用 JSON Lines 结构化写入 (非纯 print), 每行一个 dict
  2. register_gradient_hooks(): 自动给模型每层注册 backward hook,
     记录梯度 L2 范数 → 运行时可直观看到哪层梯度爆炸/消失
"""
import time, os, shutil, json
from walpurgis_ported_v9 import _dbg

_TAG = "log"


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        print('[%0.8fs] %s' % (elapsed, func.__name__))
        return result
    return clocked


class TrainLogger:
    def __init__(self, model_name, dataset):
        self.base_path = 'log/'
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.log_dir = self.base_path + cur_time
        os.makedirs(self.log_dir, exist_ok=True)
        shutil.copytree('models', self.log_dir + "/models", dirs_exist_ok=True)
        shutil.copytree('configs', self.log_dir + "/configs", dirs_exist_ok=True)
        shutil.copyfile('main.py', self.log_dir + "/main.py")
        try:
            shutil.copyfile('output/' + model_name + "_" + dataset + ".pt",
                            self.log_dir + "/" + model_name + "_" + dataset + ".pt")
        except FileNotFoundError:
            pass

        # v9: JSON Lines 日志文件
        self._jsonl_path = os.path.join(self.log_dir, "train_log.jsonl")
        self._jsonl_fh = open(self._jsonl_path, "a")
        _dbg(_TAG, f"JSONL log → {self._jsonl_path}")

    def log_json(self, record: dict):
        """追加一条结构化记录到 JSONL 文件."""
        record["_ts"] = time.time()
        self._jsonl_fh.write(json.dumps(record, default=str) + "\n")
        self._jsonl_fh.flush()

    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)
        self.log_json({"event": "model_args",
                       "args": {k: str(v) for k, v in model_args.items() if k not in ban}})

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
        self.log_json({"event": "optim_args",
                       "args": {k: str(v) for k, v in optim_args.items() if k not in ban}})


# ───── v9: 自动梯度范数监控 hook ─────

def register_gradient_hooks(model, logger: TrainLogger = None):
    """给模型每个有参数的层注册 backward hook, 打印/记录梯度 L2 范数."""
    import torch

    def _make_hook(layer_name):
        def _hook(module, grad_input, grad_output):
            for idx, g in enumerate(grad_output):
                if g is not None:
                    gnorm = g.detach().norm(2).item()
                    _dbg(_TAG, f"∇ {layer_name}  out[{idx}] L2={gnorm:.6g}")
                    if logger is not None:
                        logger.log_json({"event": "grad_norm",
                                         "layer": layer_name,
                                         "idx": idx,
                                         "l2": gnorm})
        return _hook

    handles = []
    for name, mod in model.named_modules():
        if list(mod.parameters(recurse=False)):
            h = mod.register_full_backward_hook(_make_hook(name))
            handles.append(h)
    _dbg(_TAG, f"gradient hooks registered on {len(handles)} layers")
    return handles
