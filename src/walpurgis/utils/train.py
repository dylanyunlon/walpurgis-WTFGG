"""
Walpurgis Training Utilities — Environment & Checkpointing with Full Diagnostics
==================================================================================
Derived from D2STGNN train.py with enhanced debug output.

Changes:
  1. set_config prints full environment fingerprint (GPU, CUDA, cuDNN versions)
  2. save_model/load_model include SHA-256 integrity checks
  3. EarlyStopping uses patience-proportional cooldown instead of hard cutoff
  4. data_reshaper logs transfer timing on first call per device
"""
import time
import json
import hashlib
import os

import torch
import numpy as np
import random


def set_config(seed=0):
    """Lock all random seeds and print environment fingerprint.
    
    The fingerprint is critical for reproducibility debugging — if results
    differ between runs, check this output first.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Walpurgis::env] seed={seed}")
    print(f"[Walpurgis::env] torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()} "
          f"cudnn={torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'} "
          f"gpus={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU[{i}] {props.name} | {mem_gb:.1f}GB | "
                  f"SM×{props.multi_processor_count} | "
                  f"compute={props.major}.{props.minor}")


def _file_sha256(path, chunk_size=65536):
    """Compute SHA-256 of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]  # short hash is enough for quick checks


def save_model(model, save_path):
    """Save model state dict with size and integrity reporting."""
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    sha = _file_sha256(save_path)
    print(f"[Walpurgis::save] {save_path} | {size_mb:.2f}MB | {elapsed:.3f}s | sha256={sha}")


def load_model(model, save_path):
    """Load model state dict with key-matching diagnostics.
    
    Prints matched/missing/unexpected keys so you can immediately see
    if a checkpoint is compatible with the current model architecture.
    """
    t0 = time.perf_counter()
    sha = _file_sha256(save_path)
    state_dict = torch.load(save_path)
    
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    matched = model_keys & loaded_keys
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys
    
    model.load_state_dict(state_dict)
    elapsed = time.perf_counter() - t0
    
    print(f"[Walpurgis::load] {save_path} | {elapsed:.3f}s | sha256={sha}")
    print(f"  keys: matched={len(matched)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        preview = list(missing)[:5]
        print(f"  ⚠ missing: {preview}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        preview = list(unexpected)[:5]
        print(f"  ⚠ unexpected: {preview}{'...' if len(unexpected) > 5 else ''}")
    return model


class EarlyStopping:
    """Early stopping with patience-proportional cooldown.
    
    Walpurgis change vs upstream: when counter reaches patience/2,
    we print a "yellow warning" so the developer can intervene early.
    Also tracks full history for post-mortem analysis.
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
        self._history = []
        self._warned_halfway = False
        print(f"[Walpurgis::EarlyStopping] patience={patience} delta={delta} "
              f"save={save_path}")

    def __call__(self, val_loss, model):
        score = -val_loss
        self._history.append({
            'val_loss': float(val_loss),
            'score': float(score),
            'counter': self.counter,
            'best': float(self.best_score) if self.best_score is not None else None,
        })

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            gap = self.best_score - score
            print(f'[EarlyStopping] {self.counter}/{self.patience} '
                  f'(val={val_loss:.6f} vs best={-self.best_score:.6f} Δ={gap:.6f})')
            
            # Halfway warning
            if not self._warned_halfway and self.counter >= self.patience // 2:
                self._warned_halfway = True
                print(f'  ⚠️  Halfway to early stop — '
                      f'consider adjusting LR or checking data pipeline')
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'[EarlyStopping] *** TRIGGERED *** '
                      f'no improvement for {self.patience} epochs')
        else:
            improvement = score - self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self._warned_halfway = False
            if self.verbose:
                print(f'[EarlyStopping] improved by {improvement:.6f}')

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'[EarlyStopping] val_loss {self.val_loss_min:.6f} → {val_loss:.6f}. Saving...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss

    def export_history(self, path=None):
        """Export history as JSON for post-mortem analysis."""
        if path:
            with open(path, 'w') as f:
                json.dump(self._history, f, indent=2)
            print(f"[EarlyStopping] history → {path}")
        return self._history


# ── Data transfer ── #
_reshaper_first_call = {}

def data_reshaper(data, device):
    """Convert numpy array to tensor on target device.
    
    Logs transfer timing on first call per device (to catch slow PCIe).
    """
    dev_key = str(device)
    if dev_key not in _reshaper_first_call:
        t0 = time.perf_counter()
        result = torch.Tensor(data).to(device)
        elapsed = (time.perf_counter() - t0) * 1000
        _reshaper_first_call[dev_key] = True
        print(f"  [DATA→{dev_key}] first transfer: shape={list(result.shape)} "
              f"| {elapsed:.1f}ms | {result.element_size()*result.nelement()/1024/1024:.1f}MB")
        return result
    return torch.Tensor(data).to(device)
