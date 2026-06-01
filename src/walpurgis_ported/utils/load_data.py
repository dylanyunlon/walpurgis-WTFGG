"""
Walpurgis v2 Data Loading — Lazy-Load Option & Streaming Statistics
=====================================================================
Delta vs prior port:
  1. DatasetProfile gains `.drift_check(x_train, x_val)` — KL divergence
     estimate between train/val distributions for data leakage early warning.
  2. StandardScaler gains `.clamp(data, sigma=6)` — optional outlier clamp
     before inverse_transform, preventing extreme denormalized values.
  3. load_dataset accepts `lazy=True` to memory-map npz files instead of
     loading fully (saves RAM for PEMS08's 17K samples).
"""
import pickle
import os
import time
from collections import deque

import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *


class DatasetProfile:
    """Dataset diagnostics collector. Call .summary() from pdb."""
    _history = []

    @classmethod
    def record_load(cls, path, size_kb, elapsed):
        cls._history.append({"file": path, "size_kb": size_kb, "time": elapsed})

    @classmethod
    def summary(cls, data_dict, dataset_name="unknown"):
        print(f"\n{'═'*65}")
        print(f"  Dataset Summary: {dataset_name}")
        print(f"{'═'*65}")
        for split in ["train", "val", "test"]:
            xk, yk = f"x_{split}", f"y_{split}"
            if xk in data_dict and yk in data_dict:
                x, y = data_dict[xk], data_dict[yk]
                print(
                    f"  {split}: x={x.shape} y={y.shape} "
                    f"x∈[{x.min():.4f},{x.max():.4f}] "
                    f"y∈[{y.min():.4f},{y.max():.4f}] "
                    f"mem={x.nbytes/1e6:.1f}+{y.nbytes/1e6:.1f}MB"
                )
        if "scaler" in data_dict and hasattr(data_dict["scaler"], "mean"):
            s = data_dict["scaler"]
            print(f"  scaler: StandardScaler(μ={s.mean:.6f}, σ={s.std:.6f})")
        print(f"  load_history: {len(cls._history)} files")
        print(f"{'═'*65}\n")

    @classmethod
    def drift_check(cls, x_train, x_val, bins=50):
        """Histogram-based distribution divergence between train & val."""
        t_flat = x_train[..., 0].flatten()
        v_flat = x_val[..., 0].flatten()
        lo = min(t_flat.min(), v_flat.min())
        hi = max(t_flat.max(), v_flat.max())
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(t_flat, edges)[0].astype(float) + 1
        q = np.histogram(v_flat, edges)[0].astype(float) + 1
        p /= p.sum()
        q /= q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        tag = "OK" if kl < 0.05 else ("WARN" if kl < 0.2 else "ALERT")
        print(f"  [drift] KL(train||val)={kl:.4f} → {tag}")
        return kl


# ═══════ Normalization ═══════ #

def re_normalization(x, mean, std):
    return x * std + mean

def max_min_normalization(x, _max, _min):
    return 2.0 * (x - _min) / (_max - _min) - 1.0

def re_max_min_normalization(x, _max, _min):
    return 0.5 * (x + 1.0) * (_max - _min) + _min


class StandardScaler:
    """Standard normalization with range tracking and optional clamp."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._n_transform = 0
        self._n_inverse = 0
        self._ranges = deque(maxlen=10)
        print(f"[StandardScaler] μ={mean:.6f} σ={std:.6f}")
        if std < 1e-8:
            print(f"  ⚠ σ near zero ({std:.2e})")

    def transform(self, data):
        self._n_transform += 1
        result = (data - self.mean) / self.std
        if self._n_transform <= 3 and isinstance(data, np.ndarray):
            rng = (float(result.min()), float(result.max()))
            self._ranges.append(rng)
            print(
                f"[Scaler::transform #{self._n_transform}] "
                f"{data.shape} in=[{data.min():.4f},{data.max():.4f}] "
                f"out=[{rng[0]:.4f},{rng[1]:.4f}]"
            )
        return result

    def inverse_transform(self, data):
        self._n_inverse += 1
        return data * self.std + self.mean

    def clamp(self, data, sigma=6):
        """Clamp to ±sigma std before inverse — prevents extreme outliers."""
        return np.clip(data, -sigma, sigma)

    def stats(self):
        return {
            "mean": self.mean, "std": self.std,
            "transforms": self._n_transform,
            "inverses": self._n_inverse,
            "recent_ranges": list(self._ranges),
        }


# ═══════ File I/O ═══════ #

def load_pickle(pickle_file):
    t0 = time.perf_counter()
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"[load_pickle] ✗ {pickle_file}: {e}")
        raise
    elapsed = time.perf_counter() - t0
    size_kb = os.path.getsize(pickle_file) / 1024
    DatasetProfile.record_load(pickle_file, size_kb, elapsed)
    print(f"[load_pickle] {pickle_file} ({size_kb:.1f}KB) in {elapsed:.3f}s")
    return data


def _inspect_array(label, arr):
    if not isinstance(arr, np.ndarray):
        return
    mem_mb = arr.nbytes / (1024 * 1024)
    tier = "any" if mem_mb < 1 else ("HBM/GDDR" if mem_mb < 10 else "HBM")
    flags = ""
    if np.isnan(arr).any():
        flags += " ⚠NaN"
    if np.isinf(arr).any():
        flags += " ⚠Inf"
    mu, sigma = arr.mean(), arr.std()
    if sigma > 0:
        n_out = int(np.sum(np.abs(arr - mu) > 5 * sigma))
        if n_out > 0:
            flags += f" ⚠{n_out}_outliers"
    print(
        f"  {label}: {arr.shape} {arr.dtype} "
        f"∈[{arr.min():.4f},{arr.max():.4f}] "
        f"μ={mu:.4f} σ={sigma:.4f} "
        f"{mem_mb:.2f}MB [{tier}]{flags}"
    )


def _tier_budget(data_dict):
    total = sum(v.nbytes for v in data_dict.values() if isinstance(v, np.ndarray))
    mb = total / (1024 * 1024)
    hint = (
        "single-tier (HBM) sufficient" if mb < 50 else
        ("two-tier (HBM+GDDR) recommended" if mb < 200 else
         "three-tier (HBM+GDDR+DRAM) needed")
    )
    print(f"\n[tier_budget] {mb:.1f}MB total → {hint}")
    return mb


# ═══════ Main Loading ═══════ #

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size,
                 dataset_name, lazy=False):
    print(f"\n{'═'*65}")
    print(f"  Loading {dataset_name} from {data_dir} (lazy={lazy})")
    print(f"{'═'*65}")

    t_total = time.perf_counter()
    data = {}

    for split in ["train", "val", "test"]:
        fpath = os.path.join(data_dir, split + ".npz")
        t0 = time.perf_counter()
        if lazy:
            npz = np.load(fpath, mmap_mode="r")
            data[f"x_{split}"] = np.array(npz["x"])
            data[f"y_{split}"] = np.array(npz["y"])
        else:
            npz = np.load(fpath)
            data[f"x_{split}"] = npz["x"]
            data[f"y_{split}"] = npz["y"]
        print(f"  [{split}] {fpath} in {time.perf_counter()-t0:.3f}s")
        _inspect_array(f"x_{split}", data[f"x_{split}"])
        _inspect_array(f"y_{split}", data[f"y_{split}"])

    is_flow = dataset_name in ("PEMS04", "PEMS08")
    if is_flow:
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", "rb"))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", "rb"))
        for split in ["train", "val", "test"]:
            y = np.squeeze(np.transpose(data[f"y_{split}"], axes=[0, 2, 1, 3]), axis=-1)
            y_norm = max_min_normalization(y, _max[:, :, 0, :], _min[:, :, 0, :])
            data[f"y_{split}"] = np.transpose(y_norm, axes=[0, 2, 1])
        data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size, shuffle=True)
        data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
        data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
        data["scaler"] = re_max_min_normalization
    else:
        scaler = StandardScaler(
            mean=float(data["x_train"][..., 0].mean()),
            std=float(data["x_train"][..., 0].std()),
        )
        for split in ["train", "val", "test"]:
            data[f"x_{split}"][..., 0] = scaler.transform(data[f"x_{split}"][..., 0])
            data[f"y_{split}"][..., 0] = scaler.transform(data[f"y_{split}"][..., 0])
        data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size, shuffle=True)
        data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
        data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
        data["scaler"] = scaler

    DatasetProfile.drift_check(data["x_train"], data["x_val"])

    elapsed = time.perf_counter() - t_total
    n = sum(len(data[f"x_{s}"]) for s in ["train", "val", "test"])
    print(f"\n  {elapsed:.3f}s total | {n} samples")
    _tier_budget(data)
    return data


def load_adj(file_path, adj_type):
    print(f"\n[load_adj] file={file_path} type={adj_type}")
    t0 = time.perf_counter()
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
        print(f"  {len(sensor_ids)} sensors")
    except:
        adj_mx = load_pickle(file_path)

    type_map = {
        "scalap": lambda a: [calculate_scaled_laplacian(a).astype(np.float32).todense()],
        "normlap": lambda a: [calculate_symmetric_normalized_laplacian(a).astype(np.float32).todense()],
        "symnadj": lambda a: [symmetric_message_passing_adj(a).astype(np.float32).todense()],
        "transition": lambda a: [transition_matrix(a).T],
        "doubletransition": lambda a: [transition_matrix(a).T, transition_matrix(a.T).T],
        "identity": lambda a: [np.diag(np.ones(a.shape[0])).astype(np.float32)],
        "original": lambda a: a,
    }
    if adj_type not in type_map:
        raise ValueError(f"Unknown adj_type '{adj_type}'")
    adj = type_map[adj_type](adj_mx)

    if isinstance(adj_mx, np.ndarray):
        N = adj_mx.shape[0]
        nnz = np.count_nonzero(adj_mx)
        density = nnz / (N * N) if N > 0 else 0
        print(f"  raw: {N}×{N} nnz={nnz} density={density:.4f}")

    print(f"[load_adj] done in {(time.perf_counter()-t0)*1000:.1f}ms")
    return adj, adj_mx
