"""Cascade train utils: seeded config, EarlyStopping with plateau detection, dtype-safe reshaper.

[migrate ee38e7f98] fixed deserializing issue with old checkpoint
  上游: megatron/utils.py load_checkpoint() — torch.load 裸调用遇旧 checkpoint 时抛
  ModuleNotFoundError('No module named fp16.loss_scaler')，因模块路径已从
  fp16.loss_scaler 重命名为 megatron.fp16.loss_scaler。
  上游修法: 捕获 ModuleNotFoundError，临时注入 sys.modules 别名后重试，完事清理。

  鲁迅所言: "墙上贴着'新'字，砖头还是旧砖头——路名改了，
  人却不知道从哪条路绕回去。"
  上游的修法像是悄悄在门缝里塞了张条子: 三行 try/except，
  无日志、无审计、模块污染靠 pop() 事后擦屁股。
  Walpurgis 将此逻辑封装为 CheckpointLoader，职责三分:
    1. _resolve_state_dict: 负责加载与 sys.modules 生命周期管理，模块别名进出可审计；
    2. load_checkpoint: 外层协调，失败路径有明确退出语义而非裸 exit()；
    3. _dbg 断点全链路: LOAD_ATTEMPT / LEGACY_ALIAS_INJECT / LEGACY_LOAD_OK /
       LEGACY_ALIAS_CLEANUP / LOAD_FATAL / LOAD_OK — WALPURGIS_DEBUG=1 激活。
"""
import torch
import numpy as np
import random
import sys
import os

_CAS_DBG = os.environ.get('CASCADE_DEBUG', '0') == '1'
_WDBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg(tag: str, **kw) -> None:
    """Unified debug probe; fires when WALPURGIS_DEBUG=1 or CASCADE_DEBUG=1."""
    if _WDBG or _CAS_DBG:
        parts = " ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[WALPURGIS:{tag}@train] {parts}".rstrip(), file=sys.stderr)


def set_config(seed=0):
    r"""
    Set seed.

    seed: int
        The seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _dbg("SET_CONFIG", seed=seed)


def save_model(model, save_path):
    r"""
    save model parameters.
    """
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    r"""
    load model parameters
    """
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
    return model


# ---------------------------------------------------------------------------
# [migrate ee38e7f98] fixed deserializing issue with old checkpoint
# ---------------------------------------------------------------------------
# 上游 megatron/utils.py 用一个裸 try/except 解决新旧 checkpoint 模块路径不兼容问题:
#   try: sd = torch.load(...)
#   except ModuleNotFoundError: 注入 sys.modules 别名后重试
#   except: print + exit()
#
# 鲁迅: "补丁打在裤腰带上，风一吹就不知道补到哪里去了。"
# Walpurgis 将加载过程封装为 CheckpointLoader，三职分离，全链路可审计。
# ---------------------------------------------------------------------------

_LEGACY_ALIAS_MAP = {
    # [fix ee38e7f98] 旧 checkpoint 序列化时使用 fp16.loss_scaler 路径，
    # 新代码已迁移至 megatron.fp16.loss_scaler / walpurgis.fp16.loss_scaler。
    'fp16.loss_scaler': 'megatron.fp16.loss_scaler',
}


class CheckpointLoader:
    """封装 checkpoint 加载的完整生命周期: 首次尝试 → 旧格式回退 → 失败审计。

    与上游裸 try/except 不同，Walpurgis 将 sys.modules 别名注入/清理
    收拢于单一上下文，_dbg 断点覆盖全部决策节点，无隐式副作用残留。
    """

    def __init__(self, checkpoint_path: str, alias_map: dict | None = None):
        self.checkpoint_path = checkpoint_path
        # 允许调用方覆盖别名映射，便于测试或扩展新旧路径对
        self.alias_map: dict = alias_map if alias_map is not None else _LEGACY_ALIAS_MAP
        _dbg("CHECKPOINT_LOADER_INIT",
             path=checkpoint_path,
             alias_keys=list(self.alias_map.keys()))

    def _inject_aliases(self) -> list[str]:
        """将 alias_map 中尚未注册的模块别名注入 sys.modules。
        返回本次实际注入的 key 列表，供 cleanup 精准撤销。
        [fix ee38e7f98] 仅注入缺失项，避免覆盖已有模块。
        """
        injected = []
        for old_name, new_name in self.alias_map.items():
            if old_name not in sys.modules and new_name in sys.modules:
                sys.modules[old_name] = sys.modules[new_name]
                injected.append(old_name)
                _dbg("LEGACY_ALIAS_INJECT", alias=old_name, target=new_name)
            elif old_name not in sys.modules:
                # 目标模块本身也不存在: 注入占位符将无效，预警
                _dbg("LEGACY_ALIAS_SKIP_MISSING_TARGET",
                     alias=old_name, target=new_name)
        return injected

    def _cleanup_aliases(self, injected: list[str]) -> None:
        """撤销本次注入的全部别名，恢复 sys.modules 至注入前状态。"""
        for key in injected:
            sys.modules.pop(key, None)
            _dbg("LEGACY_ALIAS_CLEANUP", removed=key)

    def _resolve_state_dict(self) -> dict:
        """核心加载逻辑: 首次尝试 → ModuleNotFoundError 时触发旧格式回退。

        [fix ee38e7f98] 旧 checkpoint 保存时 pickle 记录了 fp16.loss_scaler
        路径，torch.load 反序列化时找不到该模块而抛 ModuleNotFoundError。
        回退策略: 注入别名 → 重试 → 清理别名，别名生命周期严格局限于本次加载。
        """
        _dbg("LOAD_ATTEMPT", path=self.checkpoint_path)
        try:
            sd = torch.load(self.checkpoint_path, map_location='cpu')
            _dbg("LOAD_OK", path=self.checkpoint_path, keys=list(sd.keys()) if isinstance(sd, dict) else type(sd).__name__)
            return sd
        except ModuleNotFoundError as exc:
            # [fix ee38e7f98] 旧 checkpoint 路径不兼容: 注入别名后重试
            _dbg("LEGACY_COMPAT_TRIGGERED", exc=str(exc))
            print(f' > deserializing using the old code structure ... (alias: {list(self.alias_map.keys())})',
                  file=sys.stderr)
            injected = self._inject_aliases()
            try:
                sd = torch.load(self.checkpoint_path, map_location='cpu')
                _dbg("LEGACY_LOAD_OK", path=self.checkpoint_path, injected=injected)
                return sd
            finally:
                # finally 保证别名无论成败都被清理，副作用不泄露
                self._cleanup_aliases(injected)

    def load(self) -> dict | None:
        """外层协调入口: 调 _resolve_state_dict，任何非 ModuleNotFoundError
        异常视为不可恢复错误，记录 _dbg 后返回 None（调用方决定是否 exit）。
        上游裸 exit() 在 Walpurgis 中提升为可测试的 None 返回。
        """
        try:
            return self._resolve_state_dict()
        except Exception as exc:  # noqa: BLE001
            _dbg("LOAD_FATAL", path=self.checkpoint_path, exc=str(exc))
            print(f'could not load the checkpoint: {self.checkpoint_path!r} — {exc}',
                  file=sys.stderr)
            return None


def load_checkpoint(checkpoint_path: str,
                    alias_map: dict | None = None,
                    exit_on_failure: bool = True) -> dict | None:
    """加载 checkpoint，内建旧格式向后兼容。

    [migrate ee38e7f98] fixed deserializing issue with old checkpoint
    对应上游 megatron/utils.py load_checkpoint() 中 torch.load 调用段。

    Parameters
    ----------
    checkpoint_path : str
        checkpoint 文件路径。
    alias_map : dict, optional
        旧模块路径 → 新模块路径映射，默认使用 _LEGACY_ALIAS_MAP
        (fp16.loss_scaler → megatron.fp16.loss_scaler)。
    exit_on_failure : bool
        True 时加载失败调用 sys.exit(1)，与上游 exit() 语义等价；
        False 时返回 None，便于单元测试。

    Returns
    -------
    dict | None
        成功时返回 state dict；exit_on_failure=False 且失败时返回 None。
    """
    _dbg("LOAD_CHECKPOINT_ENTRY", path=checkpoint_path, exit_on_failure=exit_on_failure)
    loader = CheckpointLoader(checkpoint_path, alias_map=alias_map)
    sd = loader.load()
    if sd is None and exit_on_failure:
        _dbg("LOAD_CHECKPOINT_EXIT1", path=checkpoint_path)
        sys.exit(1)
    _dbg("LOAD_CHECKPOINT_DONE", path=checkpoint_path, success=(sd is not None))
    return sd


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Cascade特有: plateau detection using rolling mean comparison."""

    def __init__(self, patience, save_path, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path
        self._recent_losses = []

    def _plateau_check(self):
        """Cascade: detect plateau by comparing recent rolling means."""
        if len(self._recent_losses) < 6:
            return False
        recent = self._recent_losses[-6:]
        first_half = np.mean(recent[:3])
        second_half = np.mean(recent[3:])
        # If improvement < 0.1%, consider it a plateau
        rel_improvement = abs(first_half - second_half) / max(abs(first_half), 1e-8)
        is_plateau = rel_improvement < 0.001
        if is_plateau:
            _dbg("EARLYSTOP_PLATEAU",
                 first_half=round(float(first_half), 6),
                 second_half=round(float(second_half), 6),
                 rel_improvement=round(float(rel_improvement), 6))
        return is_plateau

    def __call__(self, val_loss, model):
        self._recent_losses.append(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            # Cascade: accelerated stopping on plateau
            if self._plateau_check() and self.counter >= max(self.patience // 2, 1):
                print(f'EarlyStopping: plateau detected at counter {self.counter}/{self.patience}')
                self.early_stop = True
            elif self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
            else:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.save_path)
        self.val_loss_min = val_loss


def data_reshaper(data, device):
    r"""
    Description:
    -----------
    Reshape data to any models. Cascade: dtype-safe conversion.
    """
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        data = torch.as_tensor(np.asarray(data)).to(device)
    else:
        data    = torch.Tensor(data).to(device)
    return data
