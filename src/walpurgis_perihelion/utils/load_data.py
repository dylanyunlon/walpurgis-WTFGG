import pickle
import os
import numpy as np
from ..dataloader import DataLoader
from .cal_adj import *

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
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y']
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))
        y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)
        data_dict['y_train'] = np.transpose(max_min_normalization(y_train, _max[:,:,0,:], _min[:,:,0,:]), axes=[0,2,1])
        data_dict['y_val'] = np.transpose(max_min_normalization(y_val, _max[:,:,0,:], _min[:,:,0,:]), axes=[0,2,1])
        data_dict['y_test'] = np.transpose(max_min_normalization(y_test, _max[:,:,0,:], _min[:,:,0,:]), axes=[0,2,1])
        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader'] = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader'] = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler'] = re_max_min_normalization
    else:
        scaler = StandardScaler(mean=data_dict['x_train'][..., 0].mean(), std=data_dict['x_train'][..., 0].std())
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
    if adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    return adj, adj_mx
