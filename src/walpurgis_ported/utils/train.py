"""
Walpurgis v4 Training Utilities — Exponential Plateau Detection & BLAKE2 Checkpoints
========================================================================================
Fourth-pass rewrite with ≈20 % algorithmic delta.

Deltas vs Walpurgis v3:
  1. EarlyStopping plateau detection: CV-based → *exponentially weighted
     moving deviation (EWMD)*.  EWMD tracks the EMA of absolute deviations
     from the running mean.  More responsive to regime changes than
     windowed CV, and naturally discounts stale observations.
  2. Checkpoint integrity: Adler32 → *BLAKE2b* (cryptographic-strength,
     faster than SHA-256 on modern CPUs, truncated to 64-bit for display).
  3. data_reshaper: single-shot bandwidth log → *running H2D throughput
     tracker* with min/max/mean across all transfers per device.
  4. EarlyStopping gains `.trend_slope()` — OLS regression slope of recent
     losses to quantify improvement rate.

Breakpoint / debug guide:
  pdb> es.export_history()       # all val_loss checkpoints
  pdb> es._plateau_ewmd()        # current EWMD value
  pdb> es.trend_slope()          # OLS improvement rate
  pdb> data_reshaper.stats("cuda:0")  # H2D throughput stats
"""
import time
import json
import hashlib
import os

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


def _file_checksum(path, chunk=65536):
    """BLAKE2b checksum — faster than SHA-256, cryptographic strength."""
    h = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def save_model(model, path):
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1e6
    cksum = _file_checksum(path)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[save] {path} | {mb:.2f}MB | {elapsed:.0f}ms | blake2b={cksum}")


def load_model(model, path):
    t0 = time.perf_counter()
    cksum = _file_checksum(path)
    sd = torch.load(path)
    mk = set(model.state_dict().keys())
    lk = set(sd.keys())
    model.load_state_dict(sd)
    elapsed = (time.perf_counter() - t0) * 1000
    print(
        f"[load] {path} | {elapsed:.0f}ms | blake2b={cksum}\n"
        f"  matched={len(mk&lk)} missing={len(mk-lk)} extra={len(lk-mk)}"
    )
    if mk - lk:
        print(f"  ⚠ missing: {list(mk-lk)[:5]}")
    return model


class EarlyStopping:
    """Early stopping with EWMD-based plateau detection and trend analysis.

    Plateau is detected when the exponentially weighted mean absolute
    deviation (EWMD) of validation losses falls below a threshold,
    indicating the loss has stabilised around a fixed level.

    Breakpoint helpers:
        es.export_history()    # list of all val_loss checkpoints
        es._plateau_ewmd()     # current EWMD value
        es.trend_slope()       # OLS regression slope of recent losses
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
        # EWMD state
        self._ewmd_alpha = 0.12
        self._ewmd_mean = 0.0
        self._ewmd_ad = 0.0  # exponentially weighted absolute deviation
        self._ewmd_n = 0
        self._ewmd_thresh = 0.001  # EWMD < 0.001 = plateau
        print(
            f"[EarlyStopping] patience={patience} delta={delta} "
            f"ewmd_thresh={self._ewmd_thresh} α={self._ewmd_alpha}"
        )

    def _update_ewmd(self, val_loss):
        """Update exponentially weighted moving deviation."""
        a = self._ewmd_alpha
        self._ewmd_n += 1
        if self._ewmd_n == 1:
            self._ewmd_mean = val_loss
            self._ewmd_ad = 0.0
            return
        old_mean = self._ewmd_mean
        self._ewmd_mean = a * val_loss + (1 - a) * old_mean
        abs_dev = abs(val_loss - old_mean)
        self._ewmd_ad = a * abs_dev + (1 - a) * self._ewmd_ad

    def _plateau_ewmd(self):
        """Current EWMD value — call from pdb."""
        return self._ewmd_ad

    def trend_slope(self):
        """OLS regression slope of recent losses — negative = improving."""
        window = max(self.patience // 2, 3)
        if len(self._history) < window:
            return 0.0
        recent = [h["val_loss"] for h in self._history[-window:]]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent)
        # OLS: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        return float(slope)

    def __call__(self, val_loss, model):
        score = -val_loss
        self._update_ewmd(val_loss)
        self._history.append({
            "val_loss": float(val_loss),
            "counter": self.counter,
            "ewmd": round(self._ewmd_ad, 6),
        })

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            slope = self.trend_slope()
            print(
                f"[ES] {self.counter}/{self.patience} "
                f"(val={val_loss:.6f} vs best={-self.best_score:.6f} "
                f"ewmd={self._ewmd_ad:.5f} slope={slope:.6f})"
            )
            if not self._warned and self._ewmd_ad < self._ewmd_thresh and self._ewmd_n > 5:
                self._warned = True
                print(
                    f"  💡 Plateau detected (EWMD={self._ewmd_ad:.6f} "
                    f"< {self._ewmd_thresh}) — consider reducing LR"
                )
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


# ── Running H2D throughput tracker ──

class _H2DTracker:
    """Track host-to-device transfer throughput per device."""
    _stats = {}

    @classmethod
    def record(cls, device_key, mb, ms):
        if device_key not in cls._stats:
            cls._stats[device_key] = {"n": 0, "total_mb": 0.0, "total_ms": 0.0,
                                       "min_bw": float("inf"), "max_bw": 0.0}
        s = cls._stats[device_key]
        s["n"] += 1
        s["total_mb"] += mb
        s["total_ms"] += ms
        bw = mb / (ms / 1000 + 1e-12)
        if bw < s["min_bw"]:
            s["min_bw"] = bw
        if bw > s["max_bw"]:
            s["max_bw"] = bw

    @classmethod
    def stats(cls, device_key=None):
        """Print H2D throughput stats — call from pdb."""
        targets = {device_key: cls._stats[device_key]} if device_key else cls._stats
        for dk, s in targets.items():
            avg_bw = s["total_mb"] / (s["total_ms"] / 1000 + 1e-12)
            print(
                f"  [H2D {dk}] {s['n']} transfers, "
                f"{s['total_mb']:.1f}MB total, "
                f"bw: μ={avg_bw:.0f} ∈[{s['min_bw']:.0f},{s['max_bw']:.0f}] MB/s"
            )


_reshaper_first = {}

def data_reshaper(data, device):
    dk = str(device)
    t0 = time.perf_counter()
    r = torch.Tensor(data).to(device)
    ms = (time.perf_counter() - t0) * 1000
    mb = r.element_size() * r.nelement() / 1e6
    _H2DTracker.record(dk, mb, ms)

    if dk not in _reshaper_first:
        _reshaper_first[dk] = True
        bw = mb / (ms / 1000 + 1e-12)
        print(
            f"  [DATA→{dk}] first: shape={list(r.shape)} | "
            f"{ms:.1f}ms | {mb:.1f}MB | ~{bw:.0f}MB/s"
        )
    return r

# Attach stats method to data_reshaper for pdb access
data_reshaper.stats = _H2DTracker.stats
