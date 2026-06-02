"""
Training utilities: seed, save/load, early stopping, data reshape.
Ported with debug instrumentation.
"""
import torch
import numpy as np
import random
import sys

_DBG = ("--debug-train" in sys.argv)


def set_config(seed_val=0):
    """Lock all random seeds for reproducibility."""
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if _DBG:
        print(f"[DBG:train] set_config seed={seed_val}  "
              f"cudnn.deterministic={torch.backends.cudnn.deterministic}")


def save_model(net, fpath):
    """Persist model weights to disk."""
    torch.save(net.state_dict(), fpath)
    if _DBG:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[DBG:train] save_model -> {fpath}  "
              f"total_params={n_params}")


def load_model(net, fpath):
    """Restore model weights from disk."""
    state = torch.load(fpath, map_location="cpu")
    net.load_state_dict(state)
    if _DBG:
        print(f"[DBG:train] load_model <- {fpath}  "
              f"keys_loaded={len(state)}")
    return net


class EarlyStopping:
    """Halt training when validation metric stagnates."""

    def __init__(self, patience, checkpoint_path, verbose=False, min_delta=0):
        self.patience = patience
        self.verbose = verbose
        self.wait_count = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_loss = np.Inf
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

    # ---- callable interface ----
    def __call__(self, current_loss, model):
        score = -current_loss

        if self.best_score is None:
            self.best_score = score
            self._checkpoint(current_loss, model)
        elif score < self.best_score - self.min_delta:
            self.wait_count += 1
            print(f"EarlyStopping patience: {self.wait_count}/{self.patience}")
            if self.wait_count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._checkpoint(current_loss, model)
            self.wait_count = 0

    def _checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"  val_loss improved "
                  f"({self.best_val_loss:.6f} -> {val_loss:.6f}), saving …")
        save_model(model, self.checkpoint_path)
        self.best_val_loss = val_loss
        if _DBG:
            print(f"[DBG:train] EarlyStopping checkpoint saved  "
                  f"best_val_loss={val_loss:.6f}")


def data_reshaper(raw, device):
    """Numpy/list -> float tensor on *device*."""
    t = torch.Tensor(raw).to(device)
    if _DBG:
        print(f"[DBG:train] data_reshaper  shape={tuple(t.shape)}  "
              f"dtype={t.dtype}  device={t.device}  "
              f"range=[{t.min().item():.4f}, {t.max().item():.4f}]")
    return t
