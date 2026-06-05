"""
load_data — Nightfall变体
算法改写:
  1. MinMax归一化: eps-guarded (_max - _min + eps) 防除零
  2. 数据加载后NaN检测 (windowing前)
  3. split后数据泄漏断言 (train/val/test无overlap)
"""
import pickle
import os
import numpy as np
from ..dataloader import DataLoader
from .cal_adj import *
from .. import _dbg


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    eps = 1e-8
    x = 1. * (x - _min) / (_max - _min + eps)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std + 1e-8  # eps防除零

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


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
    return pickle_data


def _check_nan(data, name):
    """数据NaN/Inf检测"""
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count > 0 or inf_count > 0:
        _dbg(f"data.{name}", f"⚠ NaN={nan_count} Inf={inf_count} in {name}", "data")
    return nan_count == 0 and inf_count == 0


def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']
        # NaN检测
        _check_nan(data_dict['x_' + mode], f'x_{mode}')
        _check_nan(data_dict['y_' + mode], f'y_{mode}')
    # 数据泄漏断言
    n_train = len(data_dict['x_train'])
    n_val = len(data_dict['x_val'])
    n_test = len(data_dict['x_test'])
    _dbg("data.splits", f"train={n_train} val={n_val} test={n_test}", "data")
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
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
    else:
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std=data_dict['x_train'][..., 0].std())
        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])
        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = scaler
    return data_dict


def load_adj(file_path, adj_type):
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
        raise ValueError(f"adj type '{adj_type}' not defined")
    _dbg("load_adj.type", f"{adj_type} → {len(adj) if isinstance(adj, list) else 1} matrices", "data")
    return adj, adj_mx
