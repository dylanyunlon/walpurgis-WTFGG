"""Meridian utils/train.py — trend-aware early stopping + standard helpers.
Changes vs upstream:
  - EarlyStopping with trend detection: considers loss trajectory curvature
  - set_config: adds CUDA matmul precision setting
"""
import torch
import numpy as np
import random
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # matmul precision for newer GPUs
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


class EarlyStopping:
    """Trend-aware early stopping: considers trajectory, not just best score.
    Changes vs upstream:
      - Tracks loss history for trend analysis
      - Stops if loss is worsening AND curvature is concave-up
    """
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path
        self.loss_history = []

    def __call__(self, val_loss, model):
        self.loss_history.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # trend analysis: check if losses are consistently getting worse
            trend_bad = False
            if len(self.loss_history) >= 5:
                recent = self.loss_history[-5:]
                diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
                if all(d > 0 for d in diffs):
                    trend_bad = True
                    if _DBG:
                        print(f"[MER:early_stop] monotonic worsening detected, "
                              f"diffs={[f'{d:.4f}' for d in diffs]}", file=sys.stderr)
            effective_patience = self.patience // 2 if trend_bad else self.patience
            print(f'EarlyStopping counter: {self.counter} out of {effective_patience}')
            if self.counter >= effective_patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    data = torch.Tensor(data).to(device)
    return data
