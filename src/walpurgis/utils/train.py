import time
import json

import torch
import numpy as np
import random


def set_config(seed=0):
    """Set random seed for reproducibility across all backends.

    Walpurgis: also prints the seed and backend config for audit trail.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Walpurgis::set_config] seed={seed} cudnn.deterministic=True cudnn.benchmark=False")
    print(f"[Walpurgis::set_config] torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU[{i}] {props.name} {mem_gb:.1f}GB "
                  f"SM={props.multi_processor_count} "
                  f"major={props.major} minor={props.minor}")


def save_model(model, save_path):
    """Save model state dict with size reporting."""
    t0 = time.perf_counter()
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - t0
    # Calculate file size
    import os
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Walpurgis::save_model] saved to {save_path} ({size_mb:.2f} MB) in {elapsed:.3f}s")


def load_model(model, save_path):
    """Load model state dict with key matching diagnostics."""
    t0 = time.perf_counter()
    state_dict = torch.load(save_path)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys
    model.load_state_dict(state_dict)
    elapsed = time.perf_counter() - t0
    print(f"[Walpurgis::load_model] loaded from {save_path} in {elapsed:.3f}s")
    print(f"  matched={len(model_keys & loaded_keys)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  ⚠ missing keys: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠ unexpected keys: {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Walpurgis: enhanced with timing, history tracking, and JSON-exportable state.
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
        # Walpurgis: track history for post-analysis
        self._history = []
        print(f"[Walpurgis::EarlyStopping] init patience={patience} delta={delta} "
              f"save_path={save_path}")

    def __call__(self, val_loss, model):
        score = -val_loss
        self._history.append({
            'val_loss': float(val_loss),
            'score': float(score),
            'counter': self.counter,
            'best_score': float(self.best_score) if self.best_score is not None else None,
        })

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'[Walpurgis::EarlyStopping] counter: {self.counter}/{self.patience} '
                  f'(val_loss={val_loss:.6f} vs best={-self.best_score:.6f} Δ={self.best_score - score:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'[Walpurgis::EarlyStopping] *** TRIGGERED *** '
                      f'no improvement for {self.patience} epochs')
        else:
            improvement = score - self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.verbose:
                print(f'[Walpurgis::EarlyStopping] improved by {improvement:.6f}')

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'[Walpurgis::EarlyStopping] val_loss decreased '
                  f'({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss

    def export_history(self, path=None):
        """Export early stopping history as JSON for post-analysis."""
        if path:
            with open(path, 'w') as f:
                json.dump(self._history, f, indent=2)
            print(f"[Walpurgis::EarlyStopping] history exported to {path}")
        return self._history


def data_reshaper(data, device):
    """Reshape data to tensor on target device.

    Walpurgis: prints shape and memory estimate on first call.
    """
    data = torch.Tensor(data).to(device)
    return data
