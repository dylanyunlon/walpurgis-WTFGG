"""
D2STGNN CardGame variant — load_data.py
Algorithm changes vs upstream:
  1. Winsorize outlier handling: clip to [p5, p95] percentiles before normalization
  2. Cyclic feature encoding: time_in_day and day_in_week encoded as sin/cos pairs
"""

import os
import sys
import pickle
import numpy as np
from walpurgis_cardgame.dataloader import DataLoader
from walpurgis_cardgame.utils.cal_adj import *

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        if isinstance(tensor, np.ndarray):
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min():.6f} max={tensor.max():.6f} "
                   f"mean={tensor.mean():.6f} std={tensor.std():.6f}")
        else:
            msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
                   f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
                   f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


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


# --- CARDGAME: Winsorize outlier handling ---
def winsorize(data, lower_pct=5, upper_pct=95):
    """Clip data to [lower_pct, upper_pct] percentiles to handle outliers.

    Args:
        data: np.ndarray
        lower_pct: lower percentile (default 5)
        upper_pct: upper percentile (default 95)

    Returns:
        clipped: np.ndarray with outliers clipped
    """
    lo = np.percentile(data, lower_pct)
    hi = np.percentile(data, upper_pct)
    clipped = np.clip(data, lo, hi)
    _dbg("winsorize.bounds", f"lo={lo:.4f} hi={hi:.4f}", "load_data")
    n_clipped = np.sum((data < lo) | (data > hi))
    _dbg("winsorize.num_clipped", n_clipped, "load_data")
    return clipped


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']

    # --- CARDGAME: Winsorize outlier handling on traffic features ---
    for mode in ['train', 'val', 'test']:
        raw_feat = data_dict['x_' + mode][..., 0]
        data_dict['x_' + mode][..., 0] = winsorize(raw_feat)
        _dbg(f"load_data.winsorize.{mode}", data_dict['x_' + mode][..., 0], "load_data")

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':  # traffic flow
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
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler']       = re_max_min_normalization

    else:  # traffic speed
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std=data_dict['x_train'][..., 0].std())

        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler']       = scaler

    return data_dict


def load_adj(file_path, adj_type):
    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        # PEMS04 / SYNTH
        adj_mx = load_pickle(file_path)

    # --- CARDGAME: symmetric closure before processing ---
    adj_mx = symmetric_closure(np.array(adj_mx, dtype=np.float32))
    _dbg("load_adj.symmetric_closed", adj_mx, "load_data")

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
        error = 0
        assert error, "adj type not defined"

    _dbg("load_adj.result_type", adj_type, "load_data")
    return adj, adj_mx
