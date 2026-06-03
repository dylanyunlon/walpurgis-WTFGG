"""Dataset loading and normalisation.

Algorithm changes
-----------------
1. ``StandardScaler`` — uses Welford's online algorithm to compute
   mean/std in a single pass, numerically stable for large datasets.
2. ``load_dataset`` — inserts a data-integrity checkpoint after loading:
   checks for NaN/Inf in raw arrays, prints shape + dtype + value range
   of every split so you can catch corruption before training starts.
3. ``load_adj`` — prints the chosen adj_type and resulting matrix shape
   immediately after computation for quick verification.
"""

import pickle
import os
import numpy as np
from dataloader import DataLoader
from utils.cal_adj import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    symmetric_message_passing_adj,
    transition_matrix,
)


def re_normalization(x, mean, std):
    return x * std + mean


def max_min_normalization(x, _max, _min):
    x = (x - _min) / (_max - _min + 1e-12)   # eps guard on flat signals
    return x * 2.0 - 1.0


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.0) / 2.0
    return x * (_max - _min) + _min


class StandardScaler:
    """Welford-based online scaler (numerically stable for large N)."""

    def __init__(self, mean=None, std=None, data=None):
        if data is not None:
            # single-pass Welford
            n, running_mean, M2 = 0, 0.0, 0.0
            flat = data.flatten()
            for val in flat:
                n += 1
                delta = val - running_mean
                running_mean += delta / n
                M2 += delta * (val - running_mean)
            self.mean = running_mean
            self.std = np.sqrt(M2 / n) if n > 1 else 1.0
        else:
            self.mean = mean
            self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-12)

    def inverse_transform(self, data):
        return data * self.std + self.mean


def _integrity_check(tag, arr):
    """Print shape / dtype / range and abort on NaN or Inf."""
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    vmin, vmax = arr.min(), arr.max()
    flag = ""
    if has_nan:
        flag += " *** NaN DETECTED ***"
    if has_inf:
        flag += " *** Inf DETECTED ***"
    print(f"  [{tag}] shape={arr.shape} dtype={arr.dtype} "
          f"range=[{vmin:.4f}, {vmax:.4f}]{flag}")
    if has_nan or has_inf:
        raise RuntimeError(f"Data integrity failure in {tag}")


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def load_dataset(data_dir, batch_size, valid_batch_size,
                 test_batch_size, dataset_name):
    data = {}
    print(f"[load_dataset] Loading splits from {data_dir} ...")
    for mode in ['train', 'val', 'test']:
        npz = np.load(os.path.join(data_dir, mode + '.npz'))
        data['x_' + mode] = npz['x']
        data['y_' + mode] = npz['y']
        _integrity_check(f"x_{mode}", data['x_' + mode])
        _integrity_check(f"y_{mode}", data['y_' + mode])

    is_flow = dataset_name in ('PEMS04', 'PEMS08')

    if is_flow:
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))

        for split in ['train', 'val', 'test']:
            y = np.squeeze(
                np.transpose(data[f'y_{split}'], axes=[0, 2, 1, 3]),
                axis=-1)
            y_norm = max_min_normalization(y, _max[:, :, 0, :], _min[:, :, 0, :])
            data[f'y_{split}'] = np.transpose(y_norm, axes=[0, 2, 1])

        data['train_loader'] = DataLoader(
            data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader'] = DataLoader(
            data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoader(
            data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = re_max_min_normalization

    else:
        scaler = StandardScaler(data=data['x_train'][..., 0])
        print(f"  [scaler] mean={scaler.mean:.6f}  std={scaler.std:.6f}")

        for mode in ['train', 'val', 'test']:
            data['x_' + mode][..., 0] = scaler.transform(
                data['x_' + mode][..., 0])
            data['y_' + mode][..., 0] = scaler.transform(
                data['y_' + mode][..., 0])

        data['train_loader'] = DataLoader(
            data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader'] = DataLoader(
            data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoader(
            data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scaler

    print(f"[load_dataset] Done. train={len(data['train_loader'])} batches, "
          f"val={len(data['val_loader'])}, test={len(data['test_loader'])}")
    return data


def load_adj(file_path, adj_type):
    try:
        _, _, adj_mx = load_pickle(file_path)
    except (ValueError, TypeError):
        adj_mx = load_pickle(file_path)

    dispatch = {
        "scalap":          lambda m: [calculate_scaled_laplacian(m)
                                       .astype(np.float32).todense()],
        "normlap":         lambda m: [calculate_symmetric_normalized_laplacian(m)
                                       .astype(np.float32).todense()],
        "symnadj":         lambda m: [symmetric_message_passing_adj(m)
                                       .astype(np.float32).todense()],
        "transition":      lambda m: [transition_matrix(m).T],
        "doubletransition": lambda m: [transition_matrix(m).T,
                                       transition_matrix(m.T).T],
        "identity":        lambda m: [np.diag(np.ones(m.shape[0]))
                                       .astype(np.float32)],
        "original":        lambda m: m,
    }
    if adj_type not in dispatch:
        raise ValueError(f"Unknown adj_type '{adj_type}'")

    adj = dispatch[adj_type](adj_mx)
    # diagnostic dump
    if isinstance(adj, list):
        for i, a in enumerate(adj):
            a_np = np.asarray(a)
            print(f"  [adj {i}] shape={a_np.shape} "
                  f"nnz={np.count_nonzero(a_np)} "
                  f"range=[{a_np.min():.6f}, {a_np.max():.6f}]")
    return adj, adj_mx
