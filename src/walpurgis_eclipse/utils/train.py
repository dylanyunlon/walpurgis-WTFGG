"""Eclipse train utils: Knuth seed, trend EarlyStopping, dtype-safe reshaper."""
import torch, numpy as np, random, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def set_config(seed_val=0):
    # Knuth multiplicative hash for sub-seeds
    KNUTH = 2654435761
    s_torch = (seed_val * KNUTH) & 0xFFFFFFFF
    s_numpy = ((seed_val + 1) * KNUTH) & 0xFFFFFFFF
    s_random = ((seed_val + 2) * KNUTH) & 0xFFFFFFFF
    torch.manual_seed(s_torch); torch.cuda.manual_seed(s_torch); torch.cuda.manual_seed_all(s_torch)
    random.seed(s_random); np.random.seed(s_numpy)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if _ECL_DBG: print(f"[ECL:set_config] seed={seed_val} torch={s_torch} numpy={s_numpy} random={s_random}", file=sys.stderr)

def save_model(model, save_path): torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True)); return model

class EarlyStopping:
    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = float('inf'); self.delta = delta; self.save_path = save_path
        self._recent_losses = []

    def _trend_slope(self):
        if len(self._recent_losses) < 8: return -1.0  # not enough data
        y = np.array(self._recent_losses[-8:]); x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        if _ECL_DBG: print(f"[ECL:earlystop] trend_slope={slope:.6f} (last 8 epochs)", file=sys.stderr)
        return slope

    def __call__(self, val_loss, model):
        self._recent_losses.append(val_loss)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            slope = self._trend_slope()
            # Early trigger if trend is worsening and we're past half patience
            if slope > 0 and self.counter >= self.patience // 2:
                print(f'EarlyStopping: trend worsening (slope={slope:.6f}), triggering early at {self.counter}/{self.patience}')
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
