import pickle
import os
import numpy as np
from ..dataloader import DataLoader
from .cal_adj import *


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
    except UnicodeDecodeError:
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

    scaler = StandardScaler(mean=data_dict['x_train'][..., 0].mean(),
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
    if adj_type == 'original':
        adj = adj_mx
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        adj = adj_mx
    return adj, adj_mx
