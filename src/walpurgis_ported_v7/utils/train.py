import torch
import numpy as np
import random
import os
import sys

_DBG_TRUTIL = ("--dbg-trutil" in sys.argv)


def set_config(seed=0):
    """算法改动: 额外设 CUBLAS workspace 确定性 (PyTorch 1.8+),
    保证 matmul 的结果完全可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 算法改动: CUBLAS 确定性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # 旧版 PyTorch 没有这个 API
    if _DBG_TRUTIL:
        print(f"[DBG-TRUTIL] seed={seed}  deterministic=True  "
              f"CUBLAS_WORKSPACE_CONFIG=:4096:8")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    if _DBG_TRUTIL:
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        print(f"[DBG-TRUTIL] saved model to {save_path}  "
              f"size={size_mb:.2f}MB")


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    if _DBG_TRUTIL:
        print(f"[DBG-TRUTIL] loaded model from {save_path}")
    return model


class EarlyStopping:
    """算法改动: Plateau-aware early stopping
    除了常规 patience counter, 还跟踪最近 window 个 epoch 的
    loss 下降速率。如果速率 < min_rate 且持续 patience 个 epoch,
    提前终止 — 比纯 delta 阈值更灵敏。
    """

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

        # 算法改动: 跟踪 loss 历史来计算改善速率
        self._loss_history = []
        self._rate_window = max(patience, 5)
        self._min_improve_rate = 1e-4

    def _compute_improve_rate(self):
        """最近 window 个 epoch 的平均每-epoch 改善量"""
        if len(self._loss_history) < 2:
            return float('inf')
        recent = self._loss_history[-self._rate_window:]
        if len(recent) < 2:
            return float('inf')
        # 线性拟合斜率
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        return -slope  # 正值 = loss 在下降

    def __call__(self, val_loss, model):
        self._loss_history.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1

            # 算法改动: 同时检查改善速率
            rate = self._compute_improve_rate()
            stagnant = rate < self._min_improve_rate

            if _DBG_TRUTIL:
                print(f"[DBG-TRUTIL] ES counter={self.counter}/{self.patience}  "
                      f"improve_rate={rate:.6f}  stagnant={stagnant}")

            print(f'EarlyStopping counter: {self.counter} out of '
                  f'{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                  f'Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    data = torch.Tensor(data).to(device)
    return data
