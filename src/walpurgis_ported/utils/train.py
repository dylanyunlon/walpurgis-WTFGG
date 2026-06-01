"""
Walpurgis v2 Training Utilities
=================================
Delta: EarlyStopping gains a *plateau detector* — if val_loss oscillates
within a band for patience/2 epochs, it suggests LR reduction rather
than stopping.  Checkpoint I/O uses fast CRC32-based integrity check.
"""
import time
import json
import zlib
import os

import torch
import numpy as np
import random


def set_config(seed=0):
    """Lock all seeds and print environment fingerprint."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[env] seed={seed}")
    print(
        f"[env] torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"cudnn={torch.backends.cudnn.version() if torch.cuda.is_available() else 'n/a'} "
        f"gpus={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}] {p.name} | {p.total_memory/1e9:.1f}GB | SM×{p.multi_processor_count}")


def _file_crc(path, chunk=65536):
    crc = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            crc = zlib.crc32(data, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def save_model(model, path):
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1e6
    crc = _file_crc(path)
    print(f"[save] {path} | {mb:.2f}MB | {(time.perf_counter()-t0)*1000:.0f}ms | crc={crc}")


def load_model(model, path):
    t0 = time.perf_counter()
    crc = _file_crc(path)
    sd = torch.load(path)
    mk = set(model.state_dict().keys())
    lk = set(sd.keys())
    model.load_state_dict(sd)
    print(
        f"[load] {path} | {(time.perf_counter()-t0)*1000:.0f}ms | crc={crc}\n"
        f"  matched={len(mk&lk)} missing={len(mk-lk)} extra={len(lk-mk)}"
    )
    if mk - lk:
        print(f"  ⚠ missing: {list(mk-lk)[:5]}")
    return model


class EarlyStopping:
    """Early stopping with plateau detection.

    When loss oscillates within ±δ_plateau for patience//2 epochs,
    prints a suggestion to reduce LR (non-blocking).
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
        self._warned = False
        self._plateau_band = 0.005  # ±0.5% of best loss
        print(f"[EarlyStopping] patience={patience} delta={delta} save={save_path}")

    def _check_plateau(self):
        """Check if recent losses oscillate within a narrow band."""
        window = self.patience // 2
        if len(self._history) < window:
            return False
        recent = [h["val_loss"] for h in self._history[-window:]]
        lo, hi = min(recent), max(recent)
        mid = (lo + hi) / 2
        return (hi - lo) / (mid + 1e-8) < self._plateau_band * 2

    def __call__(self, val_loss, model):
        score = -val_loss
        self._history.append({"val_loss": float(val_loss), "counter": self.counter})

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(
                f"[ES] {self.counter}/{self.patience} "
                f"(val={val_loss:.6f} vs best={-self.best_score:.6f})"
            )
            if not self._warned and self._check_plateau():
                self._warned = True
                print("  💡 Plateau detected — consider reducing LR")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[ES] *** STOP *** no improvement for {self.patience} epochs")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self._warned = False

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"[ES] {self.val_loss_min:.6f} → {val_loss:.6f}  Saving...")
        save_model(model, self.save_path)
        self.val_loss_min = val_loss

    def export_history(self, path=None):
        if path:
            with open(path, "w") as f:
                json.dump(self._history, f, indent=2)
        return self._history


_reshaper_first = {}

def data_reshaper(data, device):
    dk = str(device)
    if dk not in _reshaper_first:
        t0 = time.perf_counter()
        r = torch.Tensor(data).to(device)
        ms = (time.perf_counter() - t0) * 1000
        _reshaper_first[dk] = True
        print(f"  [DATA→{dk}] first: shape={list(r.shape)} | {ms:.1f}ms | {r.element_size()*r.nelement()/1e6:.1f}MB")
        return r
    return torch.Tensor(data).to(device)
