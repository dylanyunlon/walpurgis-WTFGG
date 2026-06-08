"""Vortex train utils: SplitMix64 seed derivation, trend+curvature EarlyStopping, dtype-safe reshaper.
Unlike upstream (direct seed, patience-only stopping) and eclipse (Knuth hash, trend-slope stopping),
Vortex uses SplitMix64 for high-quality sub-seed derivation and adds curvature (2nd derivative)
analysis to EarlyStopping for detecting loss plateaus earlier."""
import torch, numpy as np, random, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

def _splitmix64(x):
    """SplitMix64: high-quality 64-bit hash for seed derivation."""
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

def set_config(seed_val=0):
    s_torch = _splitmix64(seed_val) & 0xFFFFFFFF
    s_numpy = _splitmix64(seed_val + 1) & 0xFFFFFFFF
    s_random = _splitmix64(seed_val + 2) & 0xFFFFFFFF
    torch.manual_seed(s_torch); torch.cuda.manual_seed(s_torch); torch.cuda.manual_seed_all(s_torch)
    random.seed(s_random); np.random.seed(s_numpy)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if _VX_DBG:
        print(f"[VX:set_config@train] seed={seed_val} torch={s_torch} "
              f"numpy={s_numpy} random={s_random}", file=sys.stderr)

def save_model(model, save_path): torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True)); return model

class EarlyStopping:
    """Vortex EarlyStopping: trend + curvature (2nd derivative) analysis.
    Unlike upstream (patience-only) and eclipse (trend slope triggering at half patience),
    Vortex monitors both loss trend slope AND curvature (2nd derivative of the loss curve).
    If the curvature is positive (loss accelerating upward) and trend is worsening,
    it triggers early. This catches both plateau and divergence patterns."""
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = float('inf'); self.delta = delta; self.save_path = save_path
        self._recent_losses = []

    def _trend_and_curvature(self):
        """Compute both slope (1st derivative) and curvature (2nd derivative) of recent losses."""
        if len(self._recent_losses) < 6: return -1.0, 0.0
        y = np.array(self._recent_losses[-8:] if len(self._recent_losses) >= 8 else self._recent_losses[-6:])
        x = np.arange(len(y))
        # Linear fit for trend
        slope = np.polyfit(x, y, 1)[0]
        # Quadratic fit for curvature
        if len(y) >= 4:
            coeffs = np.polyfit(x, y, 2)
            curvature = 2 * coeffs[0]  # 2nd derivative of ax^2+bx+c is 2a
        else:
            curvature = 0.0
        if _VX_DBG:
            print(f"[VX:earlystop@train] slope={slope:.6f} curvature={curvature:.6f} "
                  f"window={len(y)}", file=sys.stderr)
        return slope, curvature

    def __call__(self, val_loss, model):
        self._recent_losses.append(val_loss)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            slope, curvature = self._trend_and_curvature()
            # Trigger if: (1) trend worsening AND curvature positive (accelerating upward)
            # AND we're past 1/3 patience
            if slope > 0 and curvature > 0 and self.counter >= max(self.patience // 3, 1):
                print(f'EarlyStopping: diverging (slope={slope:.6f}, curvature={curvature:.6f}), '
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
