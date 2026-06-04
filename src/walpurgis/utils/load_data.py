import pickle
import os
import numpy as np
from walpurgis import _dbg
from dataloader import DataLoader
from utils.cal_adj import (
    calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian,
    symmetric_message_passing_adj, transition_matrix,
    _rbf_kernel, _knn_sparsify, _symmetric_closure
)

import torch
_TAG = "data"


def re_normalization(x, mean, std):
    return x * std + mean


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        # 改动3: eps 防零 std — upstream 直接除 std, std=0 时爆炸
        self.std = std if std > 1e-8 else 1.0
        if std <= 1e-8:
            print(f"[walpurgis StandardScaler] WARNING: std={std} too small, "
                  f"clamped to 1.0")

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


# ---- 改动1: Tukey fences 异常值剔除 ----
# upstream 直接用原始数据, 不做 outlier 处理
# Tukey: [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 外的值 clip 到边界
def _tukey_clip(data, feat_idx=0):
    """对 data[..., feat_idx] 做 Tukey fences clip."""
    vals = data[..., feat_idx].flatten()
    vals_clean = vals[~np.isnan(vals)]
    if len(vals_clean) == 0:
        return data
    q1 = np.percentile(vals_clean, 25)
    q3 = np.percentile(vals_clean, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before_clip = np.sum((vals_clean < lower) | (vals_clean > upper))
    data[..., feat_idx] = np.clip(data[..., feat_idx], lower, upper)
    print(f"[walpurgis Tukey] Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
          f"bounds=[{lower:.2f}, {upper:.2f}], clipped={before_clip} samples")
    return data


# ---- 改动2: 周期性 sin/cos 编码 ----
# upstream 直接用 time_of_day / day_of_week 的原始值(0~1 或 0~6)
# 这里额外拼接 sin/cos 编码, 让模型感知周期连续性
def _add_periodic_encoding(data):
    """给 data[..., 1] (ToD) 和 data[..., 2] (DoW) 加 sin/cos."""
    if data.shape[-1] < 3:
        return data  # 不够列就跳过
    tod = data[..., 1:2]  # time of day, 0~1
    dow = data[..., 2:3]  # day of week, 0~6

    tod_sin = np.sin(2 * np.pi * tod)
    tod_cos = np.cos(2 * np.pi * tod)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # 拼到最后
    data = np.concatenate([data, tod_sin, tod_cos, dow_sin, dow_cos], axis=-1)
    print(f"[walpurgis periodic] Added sin/cos encoding, "
          f"new feat dim: {data.shape[-1]}")
    return data


def load_dataset(data_dir, batch_size, valid_batch_size,
                 test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']

    # 改动1: Tukey fences 异常值处理
    for mode in ['train', 'val', 'test']:
        data_dict['x_' + mode] = _tukey_clip(data_dict['x_' + mode], feat_idx=0)

    print(f"[walpurgis] Dataset={dataset_name}, "
          f"train={data_dict['x_train'].shape}, "
          f"val={data_dict['x_val'].shape}, "
          f"test={data_dict['x_test'].shape}")

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(
            open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(
            open("datasets/" + dataset_name + "/max.pkl", 'rb'))

        y_train = np.squeeze(
            np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(
            np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(
            np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

        y_train = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train'] = np.transpose(y_train, axes=[0, 2, 1])
        y_val = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val'] = np.transpose(y_val, axes=[0, 2, 1])
        y_test = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test'] = np.transpose(y_test, axes=[0, 2, 1])

        data_dict['train_loader'] = DataLoader(
            data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(
            data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(
            data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = re_max_min_normalization

    else:
        scaler = StandardScaler(
            mean=data_dict['x_train'][..., 0].mean(),
            std=data_dict['x_train'][..., 0].std())

        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(
                data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(
                data_dict['y_' + mode][..., 0])

        data_dict['train_loader'] = DataLoader(
            data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(
            data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(
            data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = scaler

    return data_dict


def load_adj(file_path, adj_type):
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        adj_mx = load_pickle(file_path)

    # ---- 改动4: adj 预处理链 RBF → kNN → 对称闭包 ----
    # upstream 直接用原始 adj_mx 计算 transition 等
    # 先做 RBF + kNN + 对称闭包, 再送入后续
    if isinstance(adj_mx, np.ndarray) and adj_mx.ndim == 2:
        adj_processed = _rbf_kernel(adj_mx.astype(np.float64))
        adj_processed = _knn_sparsify(adj_processed)
        adj_processed = _symmetric_closure(adj_processed)
        print(f"[walpurgis load_adj] Preprocessing chain applied: "
              f"RBF → kNN(15) → symmetric closure")
    else:
        adj_processed = adj_mx

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_processed).astype(
            np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_processed).astype(
            np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_processed).astype(
            np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_processed).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_processed).T,
               transition_matrix(adj_processed.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = adj_processed
    else:
        raise ValueError(f"adj type '{adj_type}' not defined")
    return adj, adj_mx
