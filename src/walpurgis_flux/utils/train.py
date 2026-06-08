"""Flux train utils: PCG32种子生成, 梯度方向EarlyStopping, dtype-safe reshaper.
与upstream(直接seed, patience-only stopping)和vortex(SplitMix64, curvature analysis)不同,
Flux使用PCG32做种子生成, EarlyStopping加入梯度方向分析:
通过观察验证损失的移动平均方向变化来更早检测plateau."""
import torch
import numpy as np
import random
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


def _pcg32(state, inc=1442695040888963407):
    """PCG32: 高质量32位伪随机数生成器 for seed derivation."""
    state = int(state) & 0xFFFFFFFFFFFFFFFF
    old = state
    state = ((old * 6364136223846793005) + inc) & \
        0xFFFFFFFFFFFFFFFF
    xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
    rot = (old >> 59) & 0x1F
    result = ((xorshifted >> rot) |
              (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
    return result


def set_config(seed_val=0):
    s_torch = _pcg32(seed_val) & 0xFFFFFFFF
    s_numpy = _pcg32(seed_val + 1) & 0xFFFFFFFF
    s_random = _pcg32(seed_val + 2) & 0xFFFFFFFF
    torch.manual_seed(s_torch)
    torch.cuda.manual_seed(s_torch)
    torch.cuda.manual_seed_all(s_torch)
    random.seed(s_random)
    np.random.seed(s_numpy)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if _FX_DBG:
        print(f"[FX:set_config@train] seed={seed_val} "
              f"torch={s_torch} numpy={s_numpy} "
              f"random={s_random}",
              file=sys.stderr)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(
        torch.load(save_path, map_location='cpu',
                   weights_only=True))
    return model


class EarlyStopping:
    """Flux EarlyStopping: 梯度方向 + 移动平均plateau检测.
    与upstream(patience-only)和vortex(trend+curvature)不同,
    Flux用EMA平滑后的梯度方向: 如果连续M步EMA梯度为正
    (损失在上升), 且已过最小patience的1/4, 则提前停止.
    同时还检查loss变化幅度是否小于threshold(plateau)."""
    def __init__(self, patience, save_path,
                 verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path
        self._recent_losses = []
        self._ema_grad = 0.0
        self._ema_alpha = 0.3
        self._positive_grad_streak = 0

    def _check_plateau(self):
        """检查是否进入plateau: EMA梯度方向分析"""
        if len(self._recent_losses) < 4:
            return False
        # 计算最近的损失梯度(一阶差分)
        grad = (self._recent_losses[-1] -
                self._recent_losses[-2])
        # EMA平滑
        self._ema_grad = (
            self._ema_alpha * grad +
            (1 - self._ema_alpha) * self._ema_grad)
        if self._ema_grad > 0:
            self._positive_grad_streak += 1
        else:
            self._positive_grad_streak = max(
                0, self._positive_grad_streak - 1)
        # Plateau检测: 变化幅度极小
        recent = self._recent_losses[-6:]
        if len(recent) >= 4:
            amplitude = max(recent) - min(recent)
            mean_val = np.mean(recent)
            relative_amp = amplitude / (
                abs(mean_val) + 1e-8)
            if relative_amp < 0.001:
                if _FX_DBG:
                    print(f"[FX:earlystop] plateau "
                          f"detected: rel_amp="
                          f"{relative_amp:.6f}",
                          file=sys.stderr)
                return True
        # 持续上升检测
        min_streak = max(self.patience // 4, 3)
        if self._positive_grad_streak >= min_streak:
            if _FX_DBG:
                print(f"[FX:earlystop] diverging: "
                      f"positive_grad_streak="
                      f"{self._positive_grad_streak} "
                      f"ema_grad={self._ema_grad:.6f}",
                      file=sys.stderr)
            return True
        return False

    def __call__(self, val_loss, model):
        self._recent_losses.append(val_loss)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            plateau = self._check_plateau()
            if plateau and self.counter >= max(
                    self.patience // 4, 1):
                print(
                    f'EarlyStopping: plateau/diverge '
                    f'detected at {self.counter}/'
                    f'{self.patience} '
                    f'(ema_grad={self._ema_grad:.6f})')
                self.early_stop = True
            elif self.counter >= self.patience:
                self.early_stop = True
            else:
                print(f'EarlyStopping counter: '
                      f'{self.counter} out of '
                      f'{self.patience}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self._positive_grad_streak = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.val_loss_min:.6f} --> '
                  f'{val_loss:.6f})')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        data = torch.from_numpy(data).to(device)
    else:
        data = torch.Tensor(data).to(device)
    return data
