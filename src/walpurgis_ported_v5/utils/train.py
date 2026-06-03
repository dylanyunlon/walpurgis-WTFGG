import torch
import numpy as np
import random
import json
import time

# Delta vs upstream:
#   1. set_config seeds Python hash as well
#   2. EarlyStopping tracks improvement history for debug inspection
#   3. save_model includes metadata (epoch, timestamp, loss)
#   4. data_reshaper adds optional NaN check


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # ── delta 1: hash seed for reproducible set/dict ordering ──
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_path, **meta):
    """Save model state dict + optional metadata sidecar."""
    torch.save(model.state_dict(), save_path)
    # ── delta 3: metadata sidecar ──
    if meta:
        meta_path = save_path + ".meta.json"
        meta['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve."""

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta      = delta
        self.save_path  = save_path
        # ── delta 2: improvement history ──
        self._history   = []

    def __call__(self, val_loss, model):
        score = -val_loss
        self._history.append({
            "val_loss": val_loss,
            "score":    score,
            "counter":  self.counter,
        })

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Val loss ↓ ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving…')
        save_model(model, self.save_path, best_val_loss=float(val_loss))
        self.val_loss_min = val_loss

    def report(self):
        """Breakpoint helper: print improvement history."""
        for h in self._history[-10:]:
            print(h)


def data_reshaper(data, device, check_nan=False):
    data = torch.Tensor(data).to(device)
    # ── delta 4: optional NaN guard ──
    if check_nan and torch.isnan(data).any():
        print(f"\033[91m[data_reshaper] NaN in input! shape={data.shape}\033[0m")
    return data
