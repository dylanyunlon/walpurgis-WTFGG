"""
D2STGNN CardGame variant — train.py
Algorithm changes vs upstream:
  1. EarlyStopping with plateau slope detection: measures slope of recent losses
     to detect stagnation even when loss is slowly decreasing
  2. Deterministic seed derivation: seeds are derived from base seed + epoch
     for reproducible per-epoch randomness
"""

import os
import sys
import hashlib
import torch
import numpy as np
import random

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        if isinstance(tensor, (np.ndarray,)):
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min():.6f} max={tensor.max():.6f}")
        else:
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min().item():.6f} max={tensor.max().item():.6f}")
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


# --- CARDGAME: deterministic seed derivation ---
def derive_seed(base_seed, epoch):
    """Derive a deterministic seed from base_seed and epoch using SHA-256.
    This ensures reproducible per-epoch randomness without reusing seeds.

    Args:
        base_seed: int, the base random seed
        epoch: int, current epoch number

    Returns:
        derived: int, a deterministic seed for this epoch
    """
    hash_input = f"{base_seed}:{epoch}".encode('utf-8')
    digest = hashlib.sha256(hash_input).hexdigest()
    derived = int(digest[:8], 16) % (2**31)
    _dbg("derive_seed", f"base={base_seed} epoch={epoch} derived={derived}", "train")
    return derived


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
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    CARDGAME enhancement: plateau slope detection.
    In addition to the standard patience-based check, this class measures the
    slope of the last `slope_window` losses. If the slope is too flat (> slope_threshold),
    it counts as a stagnation signal even when loss is technically still decreasing.
    """

    def __init__(self, patience, save_path, verbose=False, delta=0,
                 slope_window=10, slope_threshold=-1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path

        # --- CARDGAME: plateau slope detection ---
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
        self._loss_history = []

    def _compute_slope(self):
        """Compute linear regression slope over recent losses."""
        if len(self._loss_history) < self.slope_window:
            return None
        recent = self._loss_history[-self.slope_window:]
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        # simple linear fit: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        _dbg("early_stop.slope", f"{slope:.8f} (threshold={self.slope_threshold})", "train")
        return slope

    def __call__(self, val_loss, model):
        self._loss_history.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # --- CARDGAME: also check plateau slope ---
            slope = self._compute_slope()
            if slope is not None and slope > self.slope_threshold:
                self.counter += 1  # double-count when plateau detected
                if _CG_DEBUG:
                    print(f"[CG-DBG:early_stop] plateau detected: slope={slope:.8f}, "
                          f"counter incremented to {self.counter}", file=sys.stderr)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    data = torch.Tensor(data).to(device)
    return data
