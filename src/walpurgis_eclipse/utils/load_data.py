"""Eclipse load_data: eps-guarded scaler + NaN detection."""
import pickle, os, numpy as np, sys
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

from .cal_adj import check_nan_inf, remove_nan_inf

def re_normalization(x, mean, std): return x * std + mean
def max_min_normalization(x, _max, _min): return 1. * (x - _min) / (_max - _min) * 2. - 1.
def re_max_min_normalization(x, _max, _min): return 1. * (x + 1.) / 2. * (_max - _min) + _min

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean; self.std = max(std, 1e-8)  # eps-guarded
    def transform(self, data): return (data - self.mean) / self.std
    def inverse_transform(self, data): return (data * self.std) + self.mean

def load_pickle(f):
    try:
        with open(f, 'rb') as fh: return pickle.load(fh)
    except UnicodeDecodeError:
        with open(f, 'rb') as fh: return pickle.load(fh, encoding='latin1')

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    from ..dataloader import DataLoader
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        d = np.load(os.path.join(data_dir, mode + '.npz'))
        x = d['x']; y = d['y']
        # NaN detection
        nan_x = np.isnan(x).sum(); nan_y = np.isnan(y).sum()
        if nan_x > 0 or nan_y > 0:
            print(f"[ECL:load_data] WARNING NaN in {mode}: x={nan_x} y={nan_y}", file=sys.stderr)
            x = np.nan_to_num(x, nan=0.0); y = np.nan_to_num(y, nan=0.0)
        data_dict['x_' + mode] = x; data_dict['y_' + mode] = y
        if _ECL_DBG: print(f"[ECL:load_data] {mode}: x={x.shape} y={y.shape} x_range=[{x.min():.2f},{x.max():.2f}]", file=sys.stderr)

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(os.path.join("datasets", dataset_name, "min.pkl"), 'rb'))
        _max = pickle.load(open(os.path.join("datasets", dataset_name, "max.pkl"), 'rb'))
        for mode in ['train', 'val', 'test']:
            y = np.squeeze(np.transpose(data_dict['y_' + mode], axes=[0,2,1,3]), axis=-1)
            yn = max_min_normalization(y, _max[:,:,0,:], _min[:,:,0,:])
            data_dict['y_' + mode] = np.transpose(yn, axes=[0,2,1])
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
    from .cal_adj import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, symmetric_message_passing_adj, transition_matrix
    try: sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except: adj_mx = load_pickle(file_path)
    if adj_type == "scalap": adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap": adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj": adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition": adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition": adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity": adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original': adj = adj_mx
    else: raise ValueError(f"adj type not defined: {adj_type}")
    if _ECL_DBG: print(f"[ECL:load_adj] type={adj_type} shape={adj[0].shape}", file=sys.stderr)
    return adj, adj_mx
