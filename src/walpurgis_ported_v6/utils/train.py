"""Training utilities.

Changes
-------
1. ``EarlyStopping`` — uses an exponential moving average (alpha=0.3) of
   validation loss for the patience check instead of raw per-epoch loss.
   This filters out single-epoch spikes that would otherwise waste patience.
2. ``data_reshaper`` — adds a shape-assertion guard: if the tensor does not
   have exactly 4 dims (B, L, N, C) it raises immediately with a clear
   message, instead of silently producing garbage downstream.
3. ``set_config`` — prints the effective seed and device info at startup.
"""

import torch
import numpy as np
import random


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_config] seed={seed}  "
          f"cuda_available={torch.cuda.is_available()}  "
          f"device_count={torch.cuda.device_count()}")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [save] → {save_path}  ({n_params:,} params)")


def load_model(model, save_path):
    state = torch.load(save_path, map_location='cpu')
    model.load_state_dict(state)
    print(f"  [load] ← {save_path}")
    return model


class EarlyStopping:
    """EMA-smoothed early stopping.

    Instead of comparing raw validation loss against the best seen value,
    we maintain an exponential moving average and check *that* against the
    best EMA.  This means a single noisy epoch won't burn patience.
    """

    def __init__(self, patience, save_path, verbose=False,
                 delta=0.0, ema_alpha=0.3):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path

        self._alpha = ema_alpha
        self._ema = None
        self._best_ema = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        # update EMA
        if self._ema is None:
            self._ema = val_loss
        else:
            self._ema = self._alpha * val_loss + (1 - self._alpha) * self._ema

        score = -self._ema

        if self._best_ema is None or score > self._best_ema + self.delta:
            self._best_ema = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience} '
                  f'(ema={self._ema:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'  val_loss {self.val_loss_min:.6f} → {val_loss:.6f}')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    """Convert ndarray → Tensor with a 4-dim shape guard."""
    t = torch.Tensor(data).to(device)
    if t.ndim != 4:
        raise ValueError(
            f"[data_reshaper] Expected 4-D tensor (B,L,N,C), "
            f"got shape {list(t.shape)}")
    return t
