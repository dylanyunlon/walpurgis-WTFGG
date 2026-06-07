import os
import sys
import numpy as np
import torch
import pickle

from walpurgis_solstice.utils.cal_adj import get_adjacency_matrix, calc_adj_dtw, calc_adj_correlation

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:load:{tag}] shape={list(val.shape)} mean={val.mean().item():.4f}", file=sys.stderr)
    elif isinstance(val, np.ndarray):
        print(f"[SOL:load:{tag}] shape={list(val.shape)} mean={val.mean():.4f}", file=sys.stderr)
    else:
        print(f"[SOL:load:{tag}] {val}", file=sys.stderr)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        _sdbg("scaler_init", f"mean={mean:.4f} std={std:.4f}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            return data * self.std + self.mean
        return data * self.std + self.mean


class MinMaxScaler:
    def __init__(self, _min, _max):
        self._min = _min
        self._max = _max

    def __call__(self, data, _max, _min):
        return (data + 1.) / 2. * (_max - _min) + _min


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None,
                 dataset_name='', seq_length_x=12, seq_length_y=12, **kwargs):
    from walpurgis_solstice.dataloader import DataLoaderM

    cat_dim = kwargs.get('normalizer', 'std')
    _sdbg("loading", f"dir={dataset_dir} batch={batch_size} dataset={dataset_name}")

    data_files = {}
    for cat in ['train', 'val', 'test']:
        cat_path = os.path.join(dataset_dir, cat + '.npz')
        if os.path.exists(cat_path):
            data_files[cat] = np.load(cat_path)
        else:
            _sdbg("warn", f"missing {cat_path}")
            return None

    x_train = data_files['train']['x']
    y_train = data_files['train']['y']
    x_val = data_files['val']['x']
    y_val = data_files['val']['y']
    x_test = data_files['test']['x']
    y_test = data_files['test']['y']

    _sdbg("x_train", x_train)
    _sdbg("y_train", y_train)

    if cat_dim == 'minmax':
        _max = np.max(x_train[..., 0])
        _min = np.min(x_train[..., 0])
        scaler = MinMaxScaler(_min, _max)
        x_train[..., 0] = (x_train[..., 0] - _min) / (_max - _min) * 2. - 1.
        x_val[..., 0] = (x_val[..., 0] - _min) / (_max - _min) * 2. - 1.
        x_test[..., 0] = (x_test[..., 0] - _min) / (_max - _min) * 2. - 1.
        y_train[..., 0] = (y_train[..., 0] - _min) / (_max - _min) * 2. - 1.
        y_val[..., 0] = (y_val[..., 0] - _min) / (_max - _min) * 2. - 1.
        y_test[..., 0] = (y_test[..., 0] - _min) / (_max - _min) * 2. - 1.
    else:
        mean = x_train[..., 0].mean()
        std = x_train[..., 0].std()
        scaler = StandardScaler(mean=mean, std=std)
        x_train[..., 0] = scaler.transform(x_train[..., 0])
        y_train[..., 0] = scaler.transform(y_train[..., 0])
        x_val[..., 0] = scaler.transform(x_val[..., 0])
        y_val[..., 0] = scaler.transform(y_val[..., 0])
        x_test[..., 0] = scaler.transform(x_test[..., 0])
        y_test[..., 0] = scaler.transform(y_test[..., 0])

    data = {}
    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['train_loader'] = DataLoaderM(x_train, y_train, batch_size, shuffle=True)
    data['val_loader'] = DataLoaderM(x_val, y_val, valid_batch_size or batch_size, shuffle=False)
    data['test_loader'] = DataLoaderM(x_test, y_test, test_batch_size or batch_size, shuffle=False)
    data['scaler'] = scaler

    if cat_dim == 'minmax':
        data['_max'] = torch.FloatTensor([_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        data['_min'] = torch.FloatTensor([_min]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    _sdbg("loaded", f"train={x_train.shape} val={x_val.shape} test={x_test.shape}")
    return data


def load_adj(adj_path, adj_type='doubletransition', num_nodes=None):
    if adj_path.endswith('.pkl'):
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        if isinstance(adj_mx, np.ndarray):
            adj = [torch.FloatTensor(adj_mx)]
        else:
            adj = [torch.FloatTensor(adj_mx.toarray())]
    elif adj_path.endswith('.npy'):
        adj_mx = np.load(adj_path)
        adj = [torch.FloatTensor(adj_mx)]
    elif adj_path.endswith('.csv'):
        adj = [get_adjacency_matrix(adj_path, num_nodes)]
    else:
        adj = [torch.eye(num_nodes)]
    _sdbg("adj_loaded", adj[0])
    return adj
