"""Training logger — JSON-Lines variant with gradient norm hooks.

Changes
-------
* Log entries are written as JSON Lines (one dict per line) instead of
  plain print, so they can be parsed by tensorboard-like tools later.
* ``register_grad_hooks(model)`` attaches backward hooks that record
  per-layer gradient L2 norm — the single most useful signal for
  diagnosing vanishing/exploding gradients in deep GNN stacks.
"""

import time
import os
import json
import shutil
import torch


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        print(f'[{elapsed:0.8f}s] {func.__name__}')
        return result
    return clocked


class TrainLogger:
    def __init__(self, model_name, dataset):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.log_dir = os.path.join('log', ts)
        os.makedirs(self.log_dir, exist_ok=True)

        # snapshot source
        for d in ('models', 'configs'):
            if os.path.isdir(d):
                shutil.copytree(d, os.path.join(self.log_dir, d),
                                dirs_exist_ok=True)
        if os.path.isfile('main.py'):
            shutil.copy('main.py', self.log_dir)

        self._jl_path = os.path.join(self.log_dir, 'train_log.jsonl')
        self._jl_fh = open(self._jl_path, 'a')

        # try to backup existing checkpoints
        for suffix in ('', '_resume'):
            ckpt = f'output/{model_name}_{dataset}{suffix}.pt'
            if os.path.isfile(ckpt):
                shutil.copy(ckpt, self.log_dir)

    def log_json(self, record: dict):
        record['_ts'] = time.time()
        self._jl_fh.write(json.dumps(record) + '\n')
        self._jl_fh.flush()

    def _tabulate(self, dic, title, ban=()):
        print(f"{'=' * 16} {title} {'=' * 16}")
        for k, v in dic.items():
            if k in ban:
                continue
            print(f'|{k:>22s}:|{str(v):>22s}|')
        print('-' * 48)

    def print_model_args(self, model_args, ban=()):
        self._tabulate(model_args, 'model args', ban)
        self.log_json({"event": "model_args",
                       "args": {k: str(v) for k, v in model_args.items()
                                if k not in ban}})

    def print_optim_args(self, optim_args, ban=()):
        self._tabulate(optim_args, 'optim args', ban)
        self.log_json({"event": "optim_args",
                       "args": {k: str(v) for k, v in optim_args.items()
                                if k not in ban}})


# ── gradient-norm diagnostic hooks ─────────────────────────────────

_grad_norms = {}      # populated by hooks, read by caller


def _make_hook(name):
    def hook(module, grad_input, grad_output):
        norms = []
        for g in grad_output:
            if g is not None:
                norms.append(g.detach().norm().item())
        if norms:
            _grad_norms[name] = max(norms)
    return hook


def register_grad_hooks(model):
    """Attach gradient-norm recorders to every leaf module.
    Call ``get_grad_norms()`` after ``loss.backward()``."""
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:    # leaf
            mod.register_full_backward_hook(_make_hook(name))


def get_grad_norms():
    """Return dict {layer_name: max_grad_norm} and reset."""
    global _grad_norms
    snapshot = dict(_grad_norms)
    _grad_norms = {}
    return snapshot


def print_grad_summary(norms, top_k=5):
    """Print the top-k layers by gradient magnitude."""
    if not norms:
        return
    ranked = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  [GradNorm top-{top_k}]", end="")
    for name, val in ranked[:top_k]:
        short = name.rsplit('.', 1)[-1] if '.' in name else name
        print(f"  {short}={val:.4f}", end="")
    print()
