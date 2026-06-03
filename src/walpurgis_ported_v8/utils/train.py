import torch
import numpy as np
import random
import sys

_DBG = ("--dbg" in sys.argv)


def _dp(tag, msg):
    """Debug print — active only with --dbg flag."""
    if _DBG:
        print(f"[DBG][{tag}] {msg}", flush=True)


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _dp("set_config", f"seed={seed}  cudnn.deterministic=True")


def save_model(model, save_path):
    state = model.state_dict()
    torch.save(state, save_path)
    _dp("save_model", f"saved to {save_path}  keys={len(state)}")


def load_model(model, save_path):
    state = torch.load(save_path)
    model.load_state_dict(state)
    _dp("load_model", f"loaded from {save_path}  keys={len(state)}")
    return model


class EarlyStopping:
    """算法改动: 双判据 early stopping
    原版: 只看 val_loss 是否连续 patience 轮没下降
    改为: 同时追踪 best loss + 连续 plateau 检测
      - plateau 检测: 如果最近 plateau_window 轮的 loss 方差 < plateau_tol,
        视为陷入平台，也触发 early stop
      - 这样既能捕捉"不再下降"也能捕捉"在一个窄区间震荡"
    """

    def __init__(self, patience, save_path, verbose=False, delta=0,
                 plateau_window=10, plateau_tol=1e-5):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        # plateau detection
        self.plateau_window = plateau_window
        self.plateau_tol = plateau_tol
        self.recent_losses = []

    def __call__(self, val_loss, model):
        score = -val_loss

        # plateau tracking
        self.recent_losses.append(val_loss)
        if len(self.recent_losses) > self.plateau_window:
            self.recent_losses.pop(0)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            _dp("EarlyStopping",
                f"no improve: {self.counter}/{self.patience}  "
                f"val_loss={val_loss:.6f}  best={-self.best_score:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        # plateau check
        if len(self.recent_losses) >= self.plateau_window:
            var = np.var(self.recent_losses)
            _dp("EarlyStopping",
                f"plateau_var={var:.8f}  tol={self.plateau_tol}")
            if var < self.plateau_tol:
                print(f"[EarlyStopping] Plateau detected "
                      f"(var={var:.2e} < tol={self.plateau_tol:.2e})")
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device, dtype=torch.float32):
    """算法改动: 强制 dtype 统一
    原版: 直接 torch.Tensor(data).to(device)，可能产生 float64
    改为: 显式转 float32 并检查 NaN
    """
    t = torch.as_tensor(data, dtype=dtype).to(device)
    if _DBG:
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        if nan_count > 0 or inf_count > 0:
            _dp("data_reshaper",
                f"WARNING: nan={nan_count} inf={inf_count} shape={list(t.shape)}")
    return t
