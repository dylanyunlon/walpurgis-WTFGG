#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pickle
import os
import numpy as np
import sys
from dataloader import DataLoader
from utils.cal_adj import *

_DBG_DATA = ("--dbg-data" in sys.argv)


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


class StandardScaler():
    """算法改动: transform/inverse_transform 里加 clamp
    防止极端值在归一化后溢出 float32 范围; 同时 std=0 的通道
    加 epsilon 保护"""

    def __init__(self, mean, std):
        self.mean = mean
        # 算法改动: std 下界保护
        self.std = np.maximum(std, 1e-8) if isinstance(std, np.ndarray) else max(std, 1e-8)

    def transform(self, data):
        normed = (data - self.mean) / self.std
        # 算法改动: clamp 防止极端 z-score
        if isinstance(normed, np.ndarray):
            normed = np.clip(normed, -10.0, 10.0)
        else:
            normed = normed.clamp(-10.0, 10.0)
        return normed

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


def _sanitize_array(arr, tag=""):
    """算法改动: 通用 NaN/Inf 清理 — 用列中位数填充而非简单置零,
    这样保持了数据分布的中心性"""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count == 0 and inf_count == 0:
        return arr
    arr = arr.copy()
    # 替换 inf 为 nan 统一处理
    arr[np.isinf(arr)] = np.nan
    # 对最后一个维度做列中位数填充
    orig_shape = arr.shape
    flat = arr.reshape(-1, orig_shape[-1])
    col_median = np.nanmedian(flat, axis=0)
    inds = np.where(np.isnan(flat))
    flat[inds] = np.take(col_median, inds[1])
    arr = flat.reshape(orig_shape)
    if _DBG_DATA:
        print(f"[DBG-DATA] _sanitize[{tag}] fixed {nan_count} NaN + "
              f"{inf_count} Inf via column-median imputation")
    return arr


def load_dataset(data_dir, batch_size, valid_batch_size,
                 test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']

    # 算法改动: 数据清理管线
    for key in list(data_dict.keys()):
        data_dict[key] = _sanitize_array(data_dict[key], tag=key)

    if _DBG_DATA:
        for key in data_dict:
            v = data_dict[key]
            print(f"[DBG-DATA] {key:12s}  shape={v.shape}  "
                  f"min={v.min():.4f}  max={v.max():.4f}  "
                  f"mean={v.mean():.4f}")

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
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

        y_train_new = max_min_normalization(
            y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train'] = np.transpose(y_train_new, axes=[0, 2, 1])
        y_val_new = max_min_normalization(
            y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val'] = np.transpose(y_val_new, axes=[0, 2, 1])
        y_test_new = max_min_normalization(
            y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test'] = np.transpose(y_test_new, axes=[0, 2, 1])

        data_dict['train_loader'] = DataLoader(
            data_dict['x_train'], data_dict['y_train'],
            batch_size, shuffle=True)
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
            data_dict['x_train'], data_dict['y_train'],
            batch_size, shuffle=True)
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

    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(
            np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(
            np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(
            np.float32).todense()]
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

    if _DBG_DATA:
        if isinstance(adj, list):
            for i, a in enumerate(adj):
                a_np = np.array(a)
                print(f"[DBG-DATA] adj[{i}] shape={a_np.shape}  "
                      f"nnz={(a_np != 0).sum()}  "
                      f"density={(a_np != 0).mean():.4f}")

    return adj, adj_mx
