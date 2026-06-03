import pickle
import os
import numpy as np
import sys
from dataloader import DataLoader
from utils.cal_adj import *

_DBG = ("--dbg" in sys.argv)


def _dp(tag, msg):
    if _DBG:
        print(f"[DBG][load_data][{tag}] {msg}", flush=True)


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
    """算法改动: transform/inverse 时打印 running 统计
    帮助观察数据缩放后的值域是否合理
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self._transform_calls = 0
        self._inv_calls = 0
        _dp("StandardScaler",
            f"init  mean={mean:.4f}  std={std:.4f}")

    def transform(self, data):
        out = (data - self.mean) / self.std
        self._transform_calls += 1
        if _DBG and self._transform_calls <= 3:
            if hasattr(data, 'shape'):
                _dp("StandardScaler.transform",
                    f"call#{self._transform_calls}  "
                    f"in=[{np.min(data):.3f},{np.max(data):.3f}]  "
                    f"out=[{np.min(out):.3f},{np.max(out):.3f}]")
        return out

    def inverse_transform(self, data):
        out = (data * self.std) + self.mean
        self._inv_calls += 1
        return out


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


def load_dataset(data_dir, batch_size, valid_batch_size,
                 test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']
        _dp("load_dataset",
            f"{mode}  x={data_dict['x_'+mode].shape}  "
            f"y={data_dict['y_'+mode].shape}  "
            f"x_range=[{data_dict['x_'+mode].min():.3f},"
            f"{data_dict['x_'+mode].max():.3f}]")

    # 数据完整性检查
    assert data_dict['x_train'].shape[-1] == data_dict['x_val'].shape[-1], \
        "train/val feature dim mismatch"
    _dp("load_dataset",
        f"feat_dim={data_dict['x_train'].shape[-1]}  "
        f"num_nodes={data_dict['x_train'].shape[2]}")

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
        _min = pickle.load(
            open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(
            open("datasets/" + dataset_name + "/max.pkl", 'rb'))

        y_train = np.squeeze(np.transpose(
            data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(
            data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(
            data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

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

    _dp("load_dataset",
        f"train_batches={len(data_dict['train_loader'])}  "
        f"val_batches={len(data_dict['val_loader'])}  "
        f"test_batches={len(data_dict['test_loader'])}")
    return data_dict


def load_adj(file_path, adj_type):
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        adj_mx = load_pickle(file_path)

    _dp("load_adj",
        f"adj shape={adj_mx.shape}  type={adj_type}  "
        f"nonzero={np.count_nonzero(adj_mx)}")

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
        adj = [transition_matrix(adj_mx).T,
               transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(
            np.float32).todense()]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx
