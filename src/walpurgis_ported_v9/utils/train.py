"""
train.py  — v9 port
====================
Upstream delta (≈20 %):
  1. set_config(seed=0) → set_config(seed_val=0)  renamed for clarity
  2. EarlyStopping smooths val_loss with EMA (α = 0.25) before comparing
  3. save_model prints SHA-256 of checkpoint bytes
  4. load_model honours map_location='cpu' by default (safe for CPU-only)
  5. data_reshaper logs tensor shape on every call when debug active
"""
import hashlib
import torch
import numpy as np
import random

from walpurgis_ported_v9 import _dbg

_TAG = "train"


# ──────────────────────── seed ────────────────────────────────────

def set_config(seed_val: int = 0):
    """Lock every known RNG source to *seed_val*."""
    _dbg(_TAG, f"set_config  seed_val={seed_val}")
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────── checkpoint I/O ──────────────────────────────

def save_model(model, checkpoint_path: str):
    """Persist *model* state-dict and log SHA-256 of the blob."""
    state = model.state_dict()
    torch.save(state, checkpoint_path)
    # v9: integrity fingerprint
    with open(checkpoint_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    _dbg(_TAG, f"save_model  → {checkpoint_path}  sha256={digest}…")


def load_model(model, checkpoint_path: str):
    """Restore *model* weights from *checkpoint_path*."""
    # v9: explicit map_location avoids device mismatch on CPU-only machines
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    _dbg(_TAG, f"load_model  ← {checkpoint_path}  params={sum(p.numel() for p in model.parameters())}")
    return model


# ──────────────────── early stopping ─────────────────────────────

class EarlyStopping:
    """
    Early-stop with **EMA-smoothed** validation loss (v9 change).

    α = 0.25: puts 75 % weight on history → more robust to single
    noisy epochs than raw val_loss comparison.
    """

    _EMA_ALPHA = 0.25

    def __init__(self, patience, checkpoint_path, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.wait_count = 0             # v9: renamed from `counter`
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self._ema_loss = None           # v9: EMA state

    def _smooth(self, raw_loss: float) -> float:
        """Apply EMA to raw loss."""
        if self._ema_loss is None:
            self._ema_loss = raw_loss
        else:
            self._ema_loss = self._EMA_ALPHA * raw_loss + (1 - self._EMA_ALPHA) * self._ema_loss
        return self._ema_loss

    def __call__(self, val_loss, model):
        smoothed = self._smooth(val_loss)
        score = -smoothed
        _dbg(_TAG, f"EarlyStopping  raw={val_loss:.6f}  ema={smoothed:.6f}  "
                    f"best={self.best_score}  wait={self.wait_count}/{self.patience}")

        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score - self.delta:
            self.wait_count += 1
            print(f'EarlyStopping counter: {self.wait_count} out of {self.patience}')
            if self.wait_count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.wait_count = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}).  Saving…')
        save_model(model, self.checkpoint_path)
        self.val_loss_min = val_loss


# ──────────────────── data reshaper ──────────────────────────────

def data_reshaper(data, device):
    """Wrap ndarray → Tensor on *device*, logging shape when debug on."""
    tensor = torch.Tensor(data).to(device)
    _dbg(_TAG, f"data_reshaper  shape={list(tensor.shape)}  "
               f"dtype={tensor.dtype}  device={tensor.device}")
    return tensor
