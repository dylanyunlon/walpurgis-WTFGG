#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Data loading utilities — walpurgis_ported_v4
Modifications:
  - StandardScaler: tracks transform/inverse_transform call count + value range
  - load_dataset: prints a consolidated shape summary table after loading
  - load_adj: uses pathlib for path handling; dumps adjacency stats
  - All loaders: injected structural debug dumps
"""
import pickle
import os
import sys
from pathlib import Path
import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *


_V4_DEBUG = True

def _dbg(tag, **kw):
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][{tag}]"]
    for k, v in kw.items():
        if isinstance(v, np.ndarray):
            parts.append(f"  {k}: shape={v.shape} dtype={v.dtype} "
                         f"range=[{np.nanmin(v):.4g}, {np.nanmax(v):.4g}]")
        else:
            parts.append(f"  {k} = {v}")
    print("\n".join(parts), file=sys.stderr)


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


class StandardScaler:
    """Standard normalization with debug instrumentation (v4).
    Tracks how many times transform/inverse are called and reports value range.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._fwd_count = 0
        self._inv_count = 0
        _dbg("StandardScaler.__init__", mean=f"{mean:.6f}", std=f"{std:.6f}")

    def transform(self, data):
        self._fwd_count += 1
        result = (data - self.mean) / self.std
        if self._fwd_count <= 6:  # only print first few to avoid spam
            _dbg(f"StandardScaler.transform (call #{self._fwd_count})",
                 input_range=f"[{np.nanmin(data):.4g}, {np.nanmax(data):.4g}]",
                 output_range=f"[{np.nanmin(result):.4g}, {np.nanmax(result):.4g}]")
        return result

    def inverse_transform(self, data):
        self._inv_count += 1
        if hasattr(data, 'cpu'):  # torch tensor
            result = (data * self.std) + self.mean
        else:
            result = (data * self.std) + self.mean
        if self._inv_count <= 6:
            _dbg(f"StandardScaler.inverse_transform (call #{self._inv_count})",
                 std=f"{self.std:.6f}", mean=f"{self.mean:.6f}")
        return result


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    _dbg("load_pickle", file=pickle_file,
         type=type(pickle_data).__name__)
    return pickle_data


def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    """Load the whole dataset with consolidated shape summary (v4)."""
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']

    # v4: consolidated shape summary table
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"[v4-DBG] Dataset '{dataset_name}' shape summary:", file=sys.stderr)
    print(f"{'split':<8} {'x_shape':<28} {'y_shape':<28}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for mode in ['train', 'val', 'test']:
        xs = data_dict['x_' + mode].shape
        ys = data_dict['y_' + mode].shape
        print(f"{mode:<8} {str(xs):<28} {str(ys):<28}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    if dataset_name in ('PEMS04', 'PEMS08'):  # traffic flow
        _min = pickle.load(open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(open("datasets/" + dataset_name + "/max.pkl", 'rb'))

        y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

        y_train_new = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train'] = np.transpose(y_train_new, axes=[0, 2, 1])
        y_val_new = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val'] = np.transpose(y_val_new, axes=[0, 2, 1])
        y_test_new = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test'] = np.transpose(y_test_new, axes=[0, 2, 1])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = re_max_min_normalization

    else:  # traffic speed
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std=data_dict['x_train'][..., 0].std()
        )
        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = scaler

    _dbg("load_dataset.DONE",
         train_batches=len(data_dict['train_loader']),
         val_batches=len(data_dict['val_loader']),
         test_batches=len(data_dict['test_loader']))
    return data_dict


def load_adj(file_path, adj_type):
    """Load and preprocess adjacency matrix (v4: pathlib + adj stats dump)."""
    fpath = Path(file_path)
    _dbg("load_adj.START", file_path=str(fpath), adj_type=adj_type, exists=fpath.exists())

    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        adj_mx = load_pickle(file_path)

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        assert False, f"adj type '{adj_type}' not defined"

    # v4: dump adjacency matrix statistics
    if isinstance(adj, list):
        for i, a in enumerate(adj):
            a_arr = np.asarray(a)
            nnz = np.count_nonzero(a_arr)
            _dbg(f"load_adj.result[{i}]",
                 shape=a_arr.shape, nnz=nnz,
                 density=f"{nnz / a_arr.size:.4f}",
                 val_range=f"[{a_arr.min():.4g}, {a_arr.max():.4g}]")
    return adj, adj_mx
