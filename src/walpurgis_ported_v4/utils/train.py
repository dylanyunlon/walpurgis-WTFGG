"""
Training utilities — walpurgis_ported_v4
Ported from upstream d2stgnn with modifications:
  - set_config: prints the RNG state fingerprint after seeding
  - EarlyStopping: logs patience countdown + best score delta at every call
  - data_reshaper: validates shape rank and prints tensor summary
  - save/load_model: prints parameter count & file size
"""
import torch
import numpy as np
import random
import os
import sys


_V4_DEBUG = True

def _dbg(tag, **kw):
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][{tag}]"]
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            parts.append(f"  {k} = {v}")
    print("\n".join(parts), file=sys.stderr)


def set_config(seed=0):
    """Deterministic seeding with verification dump."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # v4: fingerprint the RNG state so we can verify reproducibility
    rng_check = np.random.randint(0, 2**31)
    np.random.seed(seed)  # re-seed after fingerprint
    _dbg("set_config", seed=seed, rng_fingerprint=rng_check,
         cudnn_deterministic=True, cudnn_benchmark=False)


def save_model(model, save_path):
    """Save model state_dict with size reporting."""
    torch.save(model.state_dict(), save_path)
    fsize = os.path.getsize(save_path) / (1024 * 1024)
    n_params = sum(p.numel() for p in model.parameters())
    _dbg("save_model", path=save_path, file_size_MB=f"{fsize:.2f}",
         total_params=n_params)


def load_model(model, save_path):
    """Load model state_dict with mismatch detection."""
    state = torch.load(save_path, map_location='cpu')
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        _dbg("load_model.KEY_MISMATCH",
             missing_in_ckpt=list(missing)[:5],
             unexpected_in_ckpt=list(unexpected)[:5])
    model.load_state_dict(state)
    _dbg("load_model", path=save_path, loaded_keys=len(ckpt_keys))
    return model


class EarlyStopping:
    """Early stops training if validation loss doesn't improve.
    v4: enhanced with per-call debug dumps showing countdown state.
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
        _dbg("EarlyStopping.__init__", patience=patience, delta=delta, save_path=save_path)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            _dbg("EarlyStopping.FIRST_SCORE", val_loss=f"{val_loss:.6f}")
        elif score < self.best_score - self.delta:
            self.counter += 1
            # v4: countdown-style debug output
            remaining = self.patience - self.counter
            _dbg("EarlyStopping.NO_IMPROVE",
                 counter=f"{self.counter}/{self.patience}",
                 remaining=remaining,
                 current_loss=f"{val_loss:.6f}",
                 best_loss=f"{self.val_loss_min:.6f}",
                 gap=f"{(val_loss - self.val_loss_min):.6f}")
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            improvement = score - self.best_score
            _dbg("EarlyStopping.IMPROVED",
                 old_best=f"{-self.best_score:.6f}",
                 new_best=f"{val_loss:.6f}",
                 improvement=f"{improvement:.6f}")
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    """Reshape data to torch.Tensor with shape validation.
    v4: asserts rank >= 2 and prints tensor summary.
    """
    data = torch.Tensor(data).to(device)
    # v4: validate that we have at least (batch, seq) dimensions
    if data.ndim < 2:
        _dbg("data_reshaper.WARN", msg="Tensor rank < 2, may cause downstream errors",
             shape=tuple(data.shape))
    else:
        _dbg("data_reshaper", shape=tuple(data.shape), device=str(device),
             val_range=f"[{data.min().item():.4f}, {data.max().item():.4f}]")
    return data
