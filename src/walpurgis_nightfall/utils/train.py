"""
train utils — Nightfall变体
算法改写:
  1. EarlyStopping: patience在连续改善后自动延长 (奖励稳定训练)
  2. data_reshaper: 加维度断言防止静默错误
"""
import torch
import numpy as np
import random
from walpurgis_nightfall import _dbg


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model


class EarlyStopping:
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.base_patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self._consecutive_improve = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            self._consecutive_improve = 0
            _dbg("early_stop", f"counter={self.counter}/{self.patience}", "trainer")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self._consecutive_improve += 1
            # 连续5次改善就额外加patience
            if self._consecutive_improve >= 5:
                self.patience = min(self.patience + 2, self.base_patience * 2)
                _dbg("early_stop", f"patience extended to {self.patience}", "trainer")
                self._consecutive_improve = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    data = torch.Tensor(data).to(device)
    assert data.dim() >= 2, f"data_reshaper: expected >=2D, got {data.dim()}D shape={data.shape}"
    return data
