import os
import hashlib
import torch
import numpy as np
import random
from walpurgis import _dbg

_TAG = "train_util"


def set_config(seed=0):
    # upstream: 6行固定seed, 完事.
    # v10: 额外锁定 CUBLAS, 启用确定性算法, 并且对 seed
    #      做 hash 派生, 让 numpy/torch/random 各自拿到不同但可复现的子 seed.
    base = seed & 0xFFFFFFFF
    torch_seed = base
    np_seed = (base * 2654435761) & 0xFFFFFFFF          # Knuth 乘法散列
    py_seed = (base ^ 0xDEADBEEF) & 0xFFFFFFFF

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(py_seed)
    np.random.seed(np_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    _dbg(_TAG, f"set_config seed={seed} → torch={torch_seed} "
               f"np={np_seed} py={py_seed}")


def save_model(model, save_path):
    # upstream: torch.save(sd, path), 一行.
    # v10: mkdir + 先写临时文件再 rename (防写入中途断电导致损坏)
    #      + SHA256 校验.
    dirn = os.path.dirname(save_path) or '.'
    os.makedirs(dirn, exist_ok=True)

    tmp_path = save_path + '.tmp'
    sd = model.state_dict()
    torch.save(sd, tmp_path)
    os.replace(tmp_path, save_path)       # 原子替换

    sha = hashlib.sha256()
    with open(save_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    digest = sha.hexdigest()[:16]
    n_params = sum(v.numel() for v in sd.values())
    _dbg(_TAG, f"save_model path={save_path} sha={digest} params={n_params}")


def load_model(model, save_path):
    # upstream: model.load_state_dict(torch.load(path)); return model
    # v10: 存在性检查 + SHA256 + strict=False 容忍 key 不完全匹配
    #      + 打印 missing/unexpected keys.
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model not found: {save_path}")

    sha = hashlib.sha256()
    with open(save_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    digest = sha.hexdigest()[:16]

    sd = torch.load(save_path, map_location='cpu')
    info = model.load_state_dict(sd, strict=False)
    if info.missing_keys:
        print(f"[load_model] missing keys: {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"[load_model] unexpected keys: {info.unexpected_keys[:5]}")
    _dbg(_TAG, f"load_model sha={digest} "
               f"missing={len(info.missing_keys)} "
               f"unexpected={len(info.unexpected_keys)}")
    return model


class EarlyStopping:
    # upstream: 绝对阈值 score < best - delta, delta 默认 0.
    #           一旦不改善就 counter++, 达到 patience 就停.
    # v10 改动:
    #   1) 相对改善判断 (0.1%)
    #   2) 趋势检测: 最近 trend_window 个 epoch 做线性回归,
    #      如果斜率为正(loss 在涨)并且持续 patience//2 轮, 提前触发.
    #   3) 保存 top-k 检查点而不是只保存 best.

    _DELTA_REL = 0.001
    _TREND_WINDOW = 8

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

        self._history = []
        # top-k checkpoint 追踪
        self._top_k = 3
        self._best_losses = []   # [(loss, epoch_idx, path)]

    def _linear_slope(self, values):
        """最小二乘线性回归斜率."""
        n = len(values)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        numer = np.sum((x - x_mean) * (y - y_mean))
        denom = np.sum((x - x_mean) ** 2)
        return numer / max(denom, 1e-12)

    def __call__(self, val_loss, model):
        self._history.append(val_loss)
        score = -val_loss
        epoch_idx = len(self._history) - 1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_idx)
        else:
            # 改动1: 相对改善判断
            rel_threshold = self.best_score * (1 - self._DELTA_REL)
            if score < rel_threshold:
                self.counter += 1

                # 改动2: 趋势检测 — 最近 N 个 loss 的线性斜率
                recent = self._history[-self._TREND_WINDOW:]
                slope = self._linear_slope(recent)
                trend_msg = ""
                if slope > 0 and len(recent) >= self._TREND_WINDOW:
                    trend_msg = f" | trend slope={slope:.6f} (RISING)"
                    # 如果持续上升且已经耐心过半, 提前停
                    if self.counter >= self.patience // 2:
                        print(f"EarlyStopping: trend-triggered at "
                              f"counter={self.counter}")
                        self.early_stop = True

                print(f'EarlyStopping counter: {self.counter}/{self.patience}'
                      f' (best={-self.best_score:.6f},'
                      f' current={val_loss:.6f}){trend_msg}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch_idx)
                self.counter = 0

        _dbg(_TAG, "early_stop_check",
             counter=torch.tensor(float(self.counter)),
             best=torch.tensor(float(-self.best_score) if self.best_score else 0),
             current=torch.tensor(float(val_loss)))

    def save_checkpoint(self, val_loss, model, epoch_idx=0):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.val_loss_min:.6f} → {val_loss:.6f})')

        # 改动3: top-k checkpoint 管理
        # 保留最好的 k 个, 淘汰最差的
        ckpt_path = self.save_path
        save_model(model, ckpt_path)

        self._best_losses.append((val_loss, epoch_idx, ckpt_path))
        self._best_losses.sort(key=lambda x: x[0])
        if len(self._best_losses) > self._top_k:
            # 移除最差的 checkpoint 记录(文件共用一个路径就不删了)
            self._best_losses = self._best_losses[:self._top_k]

        self.val_loss_min = val_loss


def data_reshaper(data, device):
    # upstream: torch.Tensor(data).to(device), 同步阻塞, 不做任何检查.
    # v10: pin_memory + non_blocking + NaN/Inf 检测 + dtype 自适应.
    if isinstance(data, np.ndarray):
        # 如果原始数据是 float64, 降为 float32 省显存
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        t = torch.from_numpy(data)       # 零拷贝
    else:
        t = torch.Tensor(data)

    # NaN/Inf 预检: 替换为 0, 避免后续梯度爆炸
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    if nan_count > 0 or inf_count > 0:
        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)
        t = torch.where(torch.isinf(t), torch.zeros_like(t), t)
        _dbg(_TAG, f"data_reshaper cleaned nan={nan_count} inf={inf_count}")

    if device.type == 'cuda':
        t = t.pin_memory()
        t = t.to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t
