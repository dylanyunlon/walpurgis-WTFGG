import os
import hashlib
import torch
import numpy as np
import random
from walpurgis import _dbg

_TAG = "train_util"


def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 改动1: 强制确定性卷积算法 + CUBLAS workspace 配置
    # upstream 只设 cudnn.deterministic; 这里额外设 CUBLAS 环境变量
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # 老版本 PyTorch 没有此 API

    _dbg(_TAG, f"set_config seed={seed}, deterministic=True")
    print(f"[v10 set_config] seed={seed}, deterministic algorithms enabled")


def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # 改动4: 保存后计算 SHA256 校验和
    sha = hashlib.sha256()
    with open(save_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    digest = sha.hexdigest()[:16]
    print(f"[v10 save_model] {save_path} | SHA256: {digest}...")
    _dbg(_TAG, f"save_model sha256={digest}")


def load_model(model, save_path):
    # 改动4: 加载前校验文件存在性 + SHA256 日志
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"[v10] Model file not found: {save_path}")
    sha = hashlib.sha256()
    with open(save_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    digest = sha.hexdigest()[:16]
    print(f"[v10 load_model] {save_path} | SHA256: {digest}...")

    model.load_state_dict(torch.load(save_path))
    _dbg(_TAG, f"load_model sha256={digest}")
    return model


class EarlyStopping:
    # 改动2: 相对改善 δ_rel — upstream 用绝对 δ=0
    # 这里用 0.1% 相对改善: 新 loss 必须比 best*(1 - δ_rel) 更低才算改善
    _DELTA_REL = 0.001  # 0.1%

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self._history = []  # 追踪完整 loss 历史

    def __call__(self, val_loss, model):
        self._history.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        else:
            # 改动2: 相对改善判断
            # upstream: score < self.best_score - self.delta (绝对)
            # v10: 要求 score > best * (1 - δ_rel), 即 loss 降幅超过 0.1%
            rel_threshold = self.best_score * (1 - self._DELTA_REL)
            if score < rel_threshold:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter}/{self.patience} '
                      f'(best={-self.best_score:.6f}, '
                      f'current={val_loss:.6f}, '
                      f'gap={100*(val_loss+self.best_score)/abs(self.best_score):.2f}%)')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        _dbg(_TAG, "early_stop_check",
             counter=torch.tensor(float(self.counter)),
             best=torch.tensor(float(-self.best_score) if self.best_score else 0),
             current=torch.tensor(float(val_loss)))

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.val_loss_min:.6f} → {val_loss:.6f}).  Saving...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    # 改动3: pin_memory + non_blocking 异步传输
    # upstream: 直接 torch.Tensor(data).to(device), 同步阻塞
    t = torch.Tensor(data)
    if device.type == 'cuda':
        t = t.pin_memory()
        t = t.to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t
