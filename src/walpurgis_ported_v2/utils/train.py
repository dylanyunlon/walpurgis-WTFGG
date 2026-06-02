"""
Training utilities: seed control, model persistence, early stopping, data shaping.
"""

import torch
import numpy as np
import random
import sys

_DBG_TRAIN = ("--debug-train" in sys.argv) or False


def set_config(seed=0):
    """Lock all random sources to *seed* for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if _DBG_TRAIN:
        print(f"[DBG:train] set_config  seed={seed}  cudnn.deterministic=True")


def save_model(model, path):
    """Persist model state dict to *path*."""
    torch.save(model.state_dict(), path)
    if _DBG_TRAIN:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[DBG:train] save_model  path={path}  total_params={n_params}")


def load_model(model, path):
    """Restore model weights from *path*."""
    model.load_state_dict(torch.load(path))
    if _DBG_TRAIN:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[DBG:train] load_model  path={path}  total_params={n_params}")
    return model


class EarlyStopping:
    """
    Halt training when validation loss stagnates beyond *patience* epochs.
    Best model is checkpointed at *checkpoint_path* automatically.
    """

    def __init__(self, patience, checkpoint_path, verbose=False, min_delta=0.0):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.min_delta = min_delta

        self._counter = 0
        self._best_score = None
        self.early_stop = False
        self._best_val_loss = np.inf

    def __call__(self, val_loss, model):
        current_score = -val_loss

        if self._best_score is None:
            self._best_score = current_score
            self._save_checkpoint(val_loss, model)
        elif current_score < self._best_score - self.min_delta:
            self._counter += 1
            if _DBG_TRAIN:
                print(f"[DBG:train] EarlyStopping  counter={self._counter}/{self.patience}  "
                      f"val_loss={val_loss:.6f}  best={-self._best_score:.6f}")
            print(f'EarlyStopping counter: {self._counter} out of {self.patience}')
            if self._counter >= self.patience:
                self.early_stop = True
        else:
            self._best_score = current_score
            self._save_checkpoint(val_loss, model)
            self._counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self._best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.checkpoint_path)
        self._best_val_loss = val_loss


def data_reshaper(raw_data, device):
    """Move numpy / tensor data onto *device* as a float Tensor."""
    out = torch.Tensor(raw_data).to(device)
    if _DBG_TRAIN:
        print(f"[DBG:train] data_reshaper  shape={tuple(out.shape)}  "
              f"device={device}  dtype={out.dtype}  "
              f"range=[{out.min().item():.4g}, {out.max().item():.4g}]")
    return out
