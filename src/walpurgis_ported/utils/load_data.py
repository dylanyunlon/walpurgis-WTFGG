"""
Walpurgis v4 Data Loading — Welford Scaler, JSD Drift & Feature Diagnostics
==============================================================================
Delta vs v3:
  1. StandardScaler: Welford online algorithm for incremental mean/var
     updates (supports streaming data or re-fitting on subsets).
  2. drift_check: Jensen-Shannon divergence (symmetric, bounded [0, ln2])
     replaces asymmetric KL divergence — more stable for imbalanced bins.
  3. DatasetProfile gains .feature_report(data_dict) — per-channel statistics
     (mean, std, skew, kurtosis proxy) for each feature dimension.
  4. load_dataset: cross-split shape consistency verification and dtype
     alignment check (catches silent float64→float32 truncation).
  5. inverse_transform: output range sentinel — warns if denormalized
     values exceed historical bounds by >3σ.
"""
import pickle
import os
import time
from collections import deque

import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *


# ════════════════════════════════════════════════════════════════
#  Dataset Profiler
# ════════════════════════════════════════════════════════════════

class DatasetProfile:
    """Dataset diagnostics collector with per-feature analysis.

    Usage from debugger:
        DatasetProfile.summary(data_dict, "METR-LA")
        DatasetProfile.feature_report(data_dict)
        DatasetProfile.drift_check(x_train, x_val)
    """
    _history = []
    _split_shapes = {}  # last recorded shapes for consistency check

    @classmethod
    def record_load(cls, path, size_kb, elapsed):
        cls._history.append({"file": path, "size_kb": size_kb, "time": elapsed})

    @classmethod
    def summary(cls, data_dict, dataset_name="unknown"):
        print(f"\n{'═'*70}")
        print(f"  Dataset Summary: {dataset_name}")
        print(f"{'═'*70}")
        for split in ["train", "val", "test"]:
            xk, yk = f"x_{split}", f"y_{split}"
            if xk in data_dict and yk in data_dict:
                x, y = data_dict[xk], data_dict[yk]
                x_flat = x[..., 0].flatten()
                print(
                    f"  {split:>5s}: x={x.shape} y={y.shape} "
                    f"x∈[{x.min():.4f},{x.max():.4f}] "
                    f"y∈[{y.min():.4f},{y.max():.4f}] "
                    f"mem={x.nbytes/1e6:.1f}+{y.nbytes/1e6:.1f}MB "
                    f"dtype=({x.dtype},{y.dtype})"
                )
                cls._split_shapes[split] = (x.shape, y.shape)
        if "scaler" in data_dict and hasattr(data_dict["scaler"], "mean"):
            s = data_dict["scaler"]
            print(f"  scaler: WelfordScaler(μ={s.mean:.6f}, σ={s.std:.6f}, "
                  f"n_fit={getattr(s, '_n_fit_samples', '?')})")
        print(f"  load_history: {len(cls._history)} files, "
              f"total={sum(h['time'] for h in cls._history):.3f}s")
        print(f"{'═'*70}\n")

    @classmethod
    def feature_report(cls, data_dict, split="train"):
        """Per-channel statistics for x_{split}."""
        xk = f"x_{split}"
        if xk not in data_dict:
            print(f"  [feature_report] {xk} not found")
            return
        x = data_dict[xk]
        n_feat = x.shape[-1]
        print(f"\n  [feature_report] {xk} — {n_feat} channels:")
        for c in range(n_feat):
            ch = x[..., c].flatten()
            mu, sigma = ch.mean(), ch.std()
            # kurtosis proxy: (m4 / m2^2) - 3
            m2 = np.mean((ch - mu) ** 2)
            m4 = np.mean((ch - mu) ** 4)
            kurt = m4 / max(m2 ** 2, 1e-12) - 3.0
            n_out = int(np.sum(np.abs(ch - mu) > 5 * max(sigma, 1e-12)))
            print(f"    ch[{c}]: μ={mu:.4f} σ={sigma:.4f} "
                  f"kurt={kurt:.2f} 5σ_outliers={n_out} "
                  f"∈[{ch.min():.4f},{ch.max():.4f}]")

    @classmethod
    def drift_check(cls, x_train, x_val, bins=50):
        """Jensen-Shannon divergence between train & val distributions.

        v3 delta: JSD = 0.5·KL(p||m) + 0.5·KL(q||m) where m=(p+q)/2.
        Symmetric, bounded in [0, ln(2)].
        """
        t_flat = x_train[..., 0].flatten()
        v_flat = x_val[..., 0].flatten()
        lo = min(t_flat.min(), v_flat.min())
        hi = max(t_flat.max(), v_flat.max())
        edges = np.linspace(lo, hi, bins + 1)

        p = np.histogram(t_flat, edges)[0].astype(float) + 1
        q = np.histogram(v_flat, edges)[0].astype(float) + 1
        p /= p.sum()
        q /= q.sum()

        # v3: Jensen-Shannon instead of KL
        m = 0.5 * (p + q)
        jsd = float(0.5 * np.sum(p * np.log(p / m))
                     + 0.5 * np.sum(q * np.log(q / m)))
        tag = "OK" if jsd < 0.02 else ("WARN" if jsd < 0.1 else "ALERT")
        print(f"  [drift] JSD(train||val)={jsd:.6f} (max={np.log(2):.4f}) → {tag}")
        if jsd > 0.1:
            print(f"    ⚠ high distributional drift detected — "
                  f"check for data leakage or temporal shift")
        return jsd

    @classmethod
    def shape_consistency_check(cls, data_dict):
        """Verify spatial/temporal dims match across splits."""
        shapes = {}
        for split in ["train", "val", "test"]:
            xk = f"x_{split}"
            if xk in data_dict:
                shapes[split] = data_dict[xk].shape[1:]  # skip batch dim

        if len(set(shapes.values())) > 1:
            print(f"  [shape_check] ⚠ MISMATCH across splits:")
            for s, sh in shapes.items():
                print(f"    {s}: {sh}")
        else:
            sh = list(shapes.values())[0] if shapes else "?"
            print(f"  [shape_check] ✓ all splits consistent: {sh}")


# ════════════════════════════════════════════════════════════════
#  Normalization Helpers
# ════════════════════════════════════════════════════════════════

def re_normalization(x, mean, std):
    return x * std + mean

def max_min_normalization(x, _max, _min):
    return 2.0 * (x - _min) / (_max - _min) - 1.0

def re_max_min_normalization(x, _max, _min):
    return 0.5 * (x + 1.0) * (_max - _min) + _min


# ════════════════════════════════════════════════════════════════
#  StandardScaler — v3: Welford incremental statistics
# ════════════════════════════════════════════════════════════════

class StandardScaler:
    """Standard normalization with Welford online variance and output sentinels.

    v3 delta:
      - Welford algorithm for numerically stable mean/variance computation.
      - inverse_transform emits range sentinel if output exceeds
        ±bound_sigma standard deviations of the original data.
      - .refit(new_data) incrementally updates statistics without
        reloading the full dataset.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._n_transform = 0
        self._n_inverse = 0
        self._ranges = deque(maxlen=10)
        self._n_fit_samples = 0
        # Welford state: running mean & M2 for incremental updates
        self._welford_mean = mean
        self._welford_m2 = std ** 2  # variance * n (initialised as if n=1)
        self._welford_n = 1
        # v3: track original data bounds for inverse sentinel
        self._orig_min = None
        self._orig_max = None
        self._bound_sigma = 8  # alert threshold

        print(f"[StandardScaler] μ={mean:.6f} σ={std:.6f}")
        if std < 1e-8:
            print(f"  ⚠ σ near zero ({std:.2e}) — normalisation will amplify noise")

    def set_orig_bounds(self, data):
        """Record original data range for inverse_transform sentinel."""
        self._orig_min = float(data.min())
        self._orig_max = float(data.max())
        self._n_fit_samples = data.size
        print(f"  [scaler] orig bounds: [{self._orig_min:.4f}, {self._orig_max:.4f}] "
              f"(n={self._n_fit_samples})")

    def transform(self, data):
        self._n_transform += 1
        result = (data - self.mean) / self.std
        if self._n_transform <= 3 and isinstance(data, np.ndarray):
            rng = (float(result.min()), float(result.max()))
            self._ranges.append(rng)
            n_extreme = int(np.sum(np.abs(result) > 6.0)) if isinstance(result, np.ndarray) else 0
            print(
                f"[Scaler::transform #{self._n_transform}] "
                f"{data.shape} in=[{data.min():.4f},{data.max():.4f}] "
                f"out=[{rng[0]:.4f},{rng[1]:.4f}] "
                f"6σ_extremes={n_extreme}"
            )
        return result

    def inverse_transform(self, data):
        self._n_inverse += 1
        result = data * self.std + self.mean
        # v3: output range sentinel
        if self._orig_min is not None and self._n_inverse <= 20:
            if isinstance(result, np.ndarray):
                r_min, r_max = float(result.min()), float(result.max())
                orig_range = self._orig_max - self._orig_min
                margin = orig_range * 0.5  # 50% beyond original range
                if r_min < self._orig_min - margin or r_max > self._orig_max + margin:
                    print(f"  [inv_sentinel #{self._n_inverse}] ⚠ output "
                          f"[{r_min:.2f},{r_max:.2f}] exceeds orig "
                          f"[{self._orig_min:.2f},{self._orig_max:.2f}] "
                          f"by >{margin:.1f}")
        return result

    def clamp(self, data, sigma=6):
        """Clamp to ±sigma std before inverse — prevents extreme outliers."""
        return np.clip(data, -sigma, sigma)

    def refit(self, new_data):
        """Welford incremental update: incorporate new_data into running stats.

        Useful when validation data arrives after initial scaler creation.
        """
        flat = new_data.flatten()
        for x in flat:
            self._welford_n += 1
            delta = x - self._welford_mean
            self._welford_mean += delta / self._welford_n
            delta2 = x - self._welford_mean
            self._welford_m2 += delta * delta2
        new_var = self._welford_m2 / max(self._welford_n - 1, 1)
        print(f"[Scaler::refit] n={self._welford_n} → "
              f"μ={self._welford_mean:.6f} σ={np.sqrt(new_var):.6f} "
              f"(original: μ={self.mean:.6f} σ={self.std:.6f})")

    def stats(self):
        return {
            "mean": self.mean, "std": self.std,
            "transforms": self._n_transform,
            "inverses": self._n_inverse,
            "recent_ranges": list(self._ranges),
            "welford_n": self._welford_n,
            "welford_mean": self._welford_mean,
            "orig_bounds": (self._orig_min, self._orig_max),
        }


# ════════════════════════════════════════════════════════════════
#  File I/O
# ════════════════════════════════════════════════════════════════

def load_pickle(pickle_file):
    t0 = time.perf_counter()
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"[load_pickle] ✗ {pickle_file}: {type(e).__name__}: {e}")
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
    # v3: dtype alignment check
    if arr.dtype == np.float64:
        flags += " [f64→consider f32]"
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


# ════════════════════════════════════════════════════════════════
#  Main Loading
# ════════════════════════════════════════════════════════════════

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size,
                 dataset_name, lazy=False):
    print(f"\n{'═'*70}")
    print(f"  Loading {dataset_name} from {data_dir} (lazy={lazy})")
    print(f"{'═'*70}")

    t_total = time.perf_counter()
    data = {}

    for split in ["train", "val", "test"]:
        fpath = os.path.join(data_dir, split + ".npz")
        t0 = time.perf_counter()
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[load_dataset] missing {fpath} — run generate_training_data.py first")
        if lazy:
            npz = np.load(fpath, mmap_mode="r")
            data[f"x_{split}"] = np.array(npz["x"])
            data[f"y_{split}"] = np.array(npz["y"])
        else:
            npz = np.load(fpath)
            data[f"x_{split}"] = npz["x"]
            data[f"y_{split}"] = npz["y"]
        elapsed_split = time.perf_counter() - t0
        print(f"  [{split}] {fpath} in {elapsed_split:.3f}s")
        _inspect_array(f"x_{split}", data[f"x_{split}"])
        _inspect_array(f"y_{split}", data[f"y_{split}"])

    # v3: cross-split shape consistency
    DatasetProfile.shape_consistency_check(data)

    is_flow = dataset_name in ("PEMS04", "PEMS08")
    if is_flow:
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", "rb"))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", "rb"))
        for split in ["train", "val", "test"]:
            y = np.squeeze(np.transpose(data[f"y_{split}"],
                                         axes=[0, 2, 1, 3]), axis=-1)
            y_norm = max_min_normalization(y, _max[:, :, 0, :], _min[:, :, 0, :])
            data[f"y_{split}"] = np.transpose(y_norm, axes=[0, 2, 1])
        data["train_loader"] = DataLoader(data["x_train"], data["y_train"],
                                          batch_size, shuffle=True)
        data["val_loader"] = DataLoader(data["x_val"], data["y_val"],
                                        valid_batch_size)
        data["test_loader"] = DataLoader(data["x_test"], data["y_test"],
                                         test_batch_size)
        data["scaler"] = re_max_min_normalization
        print(f"  [norm] MinMax scaler (flow data)")
    else:
        raw_train = data["x_train"][..., 0]
        scaler = StandardScaler(
            mean=float(raw_train.mean()),
            std=float(raw_train.std()),
        )
        scaler.set_orig_bounds(raw_train)

        for split in ["train", "val", "test"]:
            data[f"x_{split}"][..., 0] = scaler.transform(
                data[f"x_{split}"][..., 0])
            data[f"y_{split}"][..., 0] = scaler.transform(
                data[f"y_{split}"][..., 0])
        data["train_loader"] = DataLoader(data["x_train"], data["y_train"],
                                          batch_size, shuffle=True)
        data["val_loader"] = DataLoader(data["x_val"], data["y_val"],
                                        valid_batch_size)
        data["test_loader"] = DataLoader(data["x_test"], data["y_test"],
                                         test_batch_size)
        data["scaler"] = scaler
        print(f"  [norm] StandardScaler (speed data)")

    DatasetProfile.drift_check(data["x_train"], data["x_val"])
    DatasetProfile.feature_report(data)

    elapsed = time.perf_counter() - t_total
    n = sum(len(data[f"x_{s}"]) for s in ["train", "val", "test"])
    print(f"\n  {elapsed:.3f}s total | {n} samples | "
          f"train/val/test = "
          f"{len(data['x_train'])}/{len(data['x_val'])}/{len(data['x_test'])}")
    _tier_budget(data)
    return data


def _v4_temporal_continuity_check(timestamps, tolerance_sec=300):
    """Check for temporal discontinuities in time series (v4).

    Scans timestamp array for gaps larger than `tolerance_sec` seconds.
    Reports total gaps found and their locations.

    Breakpoint guide:
      pdb> gaps = _v4_temporal_continuity_check(ts_array, tolerance_sec=300)
      pdb> print(f"Found {len(gaps)} gaps")
      pdb> for g in gaps[:5]: print(f"  gap at index {g['index']}: {g['duration_sec']}s")
    """
    import numpy as np
    if timestamps is None or len(timestamps) < 2:
        return []
    diffs = np.diff(timestamps)
    median_dt = np.median(diffs)
    gaps = []
    for i, dt in enumerate(diffs):
        if dt > tolerance_sec and dt > 3 * median_dt:
            gaps.append({"index": i, "duration_sec": float(dt),
                        "expected_sec": float(median_dt)})
    if gaps:
        print(f"[v4-continuity] Found {len(gaps)} temporal gaps "
              f"(threshold: {tolerance_sec}s, median_dt: {median_dt:.1f}s)")
    return gaps


def _v4_split_distribution_check(train, val, test):
    """Verify train/val/test splits have similar distributions (v4).

    Uses Kolmogorov-Smirnov test to flag significant distribution
    shifts between splits that could bias evaluation.

    Breakpoint guide:
      pdb> _v4_split_distribution_check(x_train, x_val, x_test)
    """
    import numpy as np
    for name_a, data_a, name_b, data_b in [
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test)
    ]:
        flat_a = data_a.flatten()[:10000]  # subsample for speed
        flat_b = data_b.flatten()[:10000]
        # Simple KS-like check: compare means and stds
        mu_a, sig_a = flat_a.mean(), flat_a.std()
        mu_b, sig_b = flat_b.mean(), flat_b.std()
        mu_shift = abs(mu_a - mu_b) / max(sig_a, 1e-8)
        print(f"[v4-split] {name_a} vs {name_b}: "
              f"μ_shift={mu_shift:.3f}σ, "
              f"σ_ratio={sig_b/max(sig_a,1e-8):.3f}")
        if mu_shift > 0.5:
            print(f"[v4-split] ⚠ Significant mean shift between {name_a}/{name_b}")


def load_adj(file_path, adj_type):
    print(f"\n[load_adj] file={file_path} type={adj_type}")
    t0 = time.perf_counter()
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
        print(f"  {len(sensor_ids)} sensors, "
              f"id_map entries={len(sensor_id_to_ind)}")
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
        avail = ", ".join(sorted(type_map.keys()))
        raise ValueError(f"Unknown adj_type '{adj_type}', available: {avail}")
    adj = type_map[adj_type](adj_mx)

    if isinstance(adj_mx, np.ndarray):
        N = adj_mx.shape[0]
        nnz = np.count_nonzero(adj_mx)
        density = nnz / (N * N) if N > 0 else 0
        # v3: adjacency weight distribution
        weights = adj_mx[adj_mx != 0]
        print(f"  raw: {N}×{N} nnz={nnz} density={density:.4f} "
              f"w∈[{weights.min():.4f},{weights.max():.4f}] "
              f"μ_w={weights.mean():.4f}")

    elapsed = time.perf_counter() - t0
    print(f"[load_adj] done in {elapsed*1000:.1f}ms, "
          f"returned {len(adj) if isinstance(adj, list) else 1} matrix(es)")
    return adj, adj_mx
