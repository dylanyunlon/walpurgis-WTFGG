"""Prism train utils: Fibonacci hash seed derivation, momentum-based EarlyStopping.
Unlike upstream (direct seed, patience-only), vortex (SplitMix64, curvature analysis),
Prism uses Fibonacci hashing for sub-seed derivation (golden ratio multiplication)
and momentum-based EarlyStopping that tracks exponential moving average of loss deltas
to detect sustained deterioration vs temporary spikes."""
import torch, numpy as np, random, sys, os
_PR_DBG = os.environ.get('PRISM_DEBUG', '0') == '1'

# Golden ratio constant for Fibonacci hashing
_PHI = 0x9E3779B97F4A7C15  # 2^64 / golden_ratio


def _fibonacci_hash(x):
    """Fibonacci hash: golden ratio multiplication for high-quality seed derivation."""
    x = (x * _PHI) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 32)) & 0xFFFFFFFFFFFFFFFF


def set_config(seed_val=0):
    s_torch = _fibonacci_hash(seed_val) & 0xFFFFFFFF
    s_numpy = _fibonacci_hash(seed_val + 1) & 0xFFFFFFFF
    s_random = _fibonacci_hash(seed_val + 2) & 0xFFFFFFFF
    torch.manual_seed(s_torch); torch.cuda.manual_seed(s_torch); torch.cuda.manual_seed_all(s_torch)
    random.seed(s_random); np.random.seed(s_numpy)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if _PR_DBG:
        print(f"[PR:set_config@train] seed={seed_val} torch={s_torch} "
              f"numpy={s_numpy} random={s_random}", file=sys.stderr)


def save_model(model, save_path): torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True)); return model


class EarlyStopping:
    """Prism EarlyStopping: momentum-based (EMA of loss deltas).
    Instead of patience-only (upstream) or curvature analysis (vortex),
    Prism tracks an exponential moving average of loss changes (deltas).
    If the EMA of deltas stays positive (loss increasing) for a sustained
    period, it triggers early. This is more robust to single-epoch spikes
    while still being responsive to genuine deterioration."""
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = float('inf'); self.delta = delta; self.save_path = save_path
        self._ema_delta = 0.0
        self._ema_alpha = 0.3  # EMA smoothing factor
        self._prev_loss = None
        self._momentum_trigger_count = 0

    def _update_momentum(self, val_loss):
        """Update EMA of loss deltas."""
        if self._prev_loss is not None:
            delta = val_loss - self._prev_loss
            self._ema_delta = (self._ema_alpha * delta +
                               (1 - self._ema_alpha) * self._ema_delta)
            if _PR_DBG:
                print(f"[PR:earlystop@train] ema_delta={self._ema_delta:.6f} "
                      f"raw_delta={delta:.6f}", file=sys.stderr)
            # Track sustained positive momentum
            if self._ema_delta > 0:
                self._momentum_trigger_count += 1
            else:
                self._momentum_trigger_count = max(0,
                    self._momentum_trigger_count - 1)
        self._prev_loss = val_loss

    def __call__(self, val_loss, model):
        self._update_momentum(val_loss)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # Prism: trigger early if momentum sustained positive for patience/2 epochs
            if (self._momentum_trigger_count >= max(self.patience // 2, 3)
                    and self.counter >= max(self.patience // 3, 1)):
                print(f'EarlyStopping: sustained deterioration '
                      f'(ema_delta={self._ema_delta:.6f}, '
                      f'momentum_count={self._momentum_trigger_count}), '
                      f'triggering at {self.counter}/{self.patience}')
                self.early_stop = True
            elif self.counter >= self.patience:
                self.early_stop = True
            else:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score = score; self.save_checkpoint(val_loss, model); self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        save_model(model, self.save_path); self.val_loss_min = val_loss


def data_reshaper(data, device):
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64: data = data.astype(np.float32)
        if np.isnan(data).any(): data = np.nan_to_num(data, nan=0.0)
        data = torch.from_numpy(data).to(device)
    else:
        data = torch.Tensor(data).to(device)
    return data
