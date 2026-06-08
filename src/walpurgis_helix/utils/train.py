"""Helix train utils: Fibonacci-hash seed derivation, momentum-based EarlyStopping, dtype-safe reshaper.
Unlike upstream (direct seed, patience-only stopping) and vortex (SplitMix64, curvature stopping),
Helix uses Fibonacci hashing for sub-seed derivation and adds momentum tracking to EarlyStopping
for detecting loss oscillation patterns earlier."""
import torch, numpy as np, random, sys, os
_HX_DBG = os.environ.get('HELIX_DEBUG', '0') == '1'

def _fibonacci_hash(x):
    """Fibonacci hashing: multiply by golden ratio inverse for uniform seed spreading."""
    GOLDEN = 0x9E3779B97F4A7C15  # floor(2^64 / phi)
    return ((x * GOLDEN) >> 32) & 0xFFFFFFFF

def set_config(seed_val=0):
    s_torch = _fibonacci_hash(seed_val)
    s_numpy = _fibonacci_hash(seed_val + 1)
    s_random = _fibonacci_hash(seed_val + 2)
    torch.manual_seed(s_torch); torch.cuda.manual_seed(s_torch); torch.cuda.manual_seed_all(s_torch)
    random.seed(s_random); np.random.seed(s_numpy)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if _HX_DBG:
        print(f"[HX:set_config@train] seed={seed_val} torch={s_torch} "
              f"numpy={s_numpy} random={s_random}", file=sys.stderr)

def save_model(model, save_path): torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True)); return model

class EarlyStopping:
    """Helix EarlyStopping: momentum-based oscillation detection.
    Unlike upstream (patience-only) and vortex (curvature analysis),
    Helix tracks loss momentum (exponential moving average of loss deltas).
    If momentum is positive (losses consistently rising) for too long, triggers early.
    This catches slow divergence patterns that patience-only would miss."""
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = float('inf'); self.delta = delta; self.save_path = save_path
        self._loss_momentum = 0.0
        self._prev_loss = None
        self._momentum_decay = 0.9

    def _update_momentum(self, val_loss):
        """Update EMA of loss deltas to detect sustained deterioration."""
        if self._prev_loss is not None:
            delta = val_loss - self._prev_loss
            self._loss_momentum = (
                self._momentum_decay * self._loss_momentum +
                (1 - self._momentum_decay) * delta)
            if _HX_DBG:
                print(f"[HX:earlystop@train] momentum={self._loss_momentum:.6f} "
                      f"delta={delta:.6f}", file=sys.stderr)
        self._prev_loss = val_loss
        return self._loss_momentum

    def __call__(self, val_loss, model):
        momentum = self._update_momentum(val_loss)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # Helix: momentum-based early trigger
            if (momentum > 0.01 and
                    self.counter >= max(self.patience // 2, 1)):
                print(f'EarlyStopping: sustained momentum={momentum:.6f}, '
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
