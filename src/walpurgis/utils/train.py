"""Cascade train utils: seeded config, EarlyStopping with plateau detection, dtype-safe reshaper."""
import torch
import numpy as np
import random
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'


def set_config(seed=0):
    r"""
    Set seed.

    seed: int
        The seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if _CAS_DBG:
        print(f"[CAS:set_config@train] seed={seed}", file=sys.stderr)


def save_model(model, save_path):
    r"""
    save model parameters.
    """
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    r"""
    load model parameters
    """
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Cascade特有: plateau detection using rolling mean comparison."""

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path
        self._recent_losses = []

    def _plateau_check(self):
        """Cascade: detect plateau by comparing recent rolling means."""
        if len(self._recent_losses) < 6:
            return False
        recent = self._recent_losses[-6:]
        first_half = np.mean(recent[:3])
        second_half = np.mean(recent[3:])
        # If improvement < 0.1%, consider it a plateau
        rel_improvement = abs(first_half - second_half) / max(abs(first_half), 1e-8)
        is_plateau = rel_improvement < 0.001
        if _CAS_DBG and is_plateau:
            print(f"[CAS:earlystop@train] plateau detected: "
                  f"first_half={first_half:.6f} second_half={second_half:.6f} "
                  f"rel_improvement={rel_improvement:.6f}", file=sys.stderr)
        return is_plateau

    def __call__(self, val_loss, model):
        self._recent_losses.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # Cascade: accelerated stopping on plateau
            if self._plateau_check() and self.counter >= max(self.patience // 2, 1):
                print(f'EarlyStopping: plateau detected at counter {self.counter}/{self.patience}')
                self.early_stop = True
            elif self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
            else:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    r"""
    Description:
    -----------
    Reshape data to any models. Cascade: dtype-safe conversion.
    """
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        data = torch.from_numpy(data).to(device)
    else:
        data    = torch.Tensor(data).to(device)
    return data
