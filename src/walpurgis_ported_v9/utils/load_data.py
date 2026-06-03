"""
load_data.py — v9 port
Algo delta:
  1. StandardScaler 用 Welford 在线算法增量计算 mean/var
     → 可处理不完全 fit 进内存的流式数据
  2. re_max_min_normalization 对 (x+1)/2 加 clamp(0,1) 防越界后再映射
  3. load_adj 用 dispatch dict 替代 if-elif 链, 且新增 'spectral' 类型
     (shift-invert ARPACK 截断 top-k 特征值做低通邻接)
  4. load_dataset 在每个 split 加载后打印 shape + 基本统计
"""
import pickle, os, math
import numpy as np
from scipy.sparse import linalg as sp_linalg, csr_matrix, eye as sp_eye
from dataloader import DataLoader
from utils.cal_adj import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    symmetric_message_passing_adj,
    transition_matrix,
)
from walpurgis_ported_v9 import _dbg

_TAG = "load_data"

# ────────── normalisation helpers ──────────

def re_normalization(x, mean, std):
    return x * std + mean

def max_min_normalization(x, _max, _min):
    x = 1.0 * (x - _min) / (_max - _min)
    x = 2.0 * x - 1.0
    return x

def re_max_min_normalization(x, _max, _min):
    # v9: clamp the (0,1) intermediate to avoid float drift exceeding bounds
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = x * (_max - _min) + _min
    return x


# ─── v9: Welford online StandardScaler ───

class StandardScaler:
    """
    Welford 增量 StandardScaler.
    upstream 一次性 .mean()/.std() → v9 用 Welford 单遍扫描,
    数值上更稳定, 且可扩展到 streaming 场景.
    """
    def __init__(self, data_1d: np.ndarray = None, mean=None, std=None):
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif data_1d is not None:
            self.mean, self.std = self._welford(data_1d)
        else:
            raise ValueError("provide either data_1d or (mean, std)")
        _dbg(_TAG, f"StandardScaler  mean={self.mean:.6g}  std={self.std:.6g}")

    @staticmethod
    def _welford(arr: np.ndarray):
        flat = arr.ravel()
        n = 0
        mu = 0.0
        m2 = 0.0
        for val in flat:
            n += 1
            delta = val - mu
            mu += delta / n
            m2 += delta * (val - mu)
        var = m2 / n if n > 1 else 0.0
        return mu, math.sqrt(var + 1e-12)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ────────── pickle loader ──────────

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


# ─── v9: spectral low-pass adj (新增 adj_type) ───

def _spectral_low_pass(adj_mx, top_k=16):
    """用 ARPACK 取 top-k 最小非零特征值构建低通近似邻接矩阵."""
    L = calculate_symmetric_normalized_laplacian(adj_mx)
    L = csr_matrix(L, dtype=np.float64)
    n = L.shape[0]
    k = min(top_k, n - 2)
    evals, evecs = sp_linalg.eigsh(L, k=k, which='SM')
    # 重构: A_approx = I - V diag(λ) V^T  (低通投影)
    approx = sp_eye(n) - evecs @ np.diag(evals) @ evecs.T
    _dbg(_TAG, f"spectral_low_pass  k={k}  eigenrange=[{evals.min():.4g}, {evals.max():.4g}]")
    return np.asarray(approx, dtype=np.float32)


# ────────── load dataset ──────────

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        arr = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode] = arr['x']
        data_dict['y_' + mode] = arr['y']
        _dbg(_TAG, f"loaded {mode}  x={arr['x'].shape}  y={arr['y'].shape}  "
                    f"x_mean={arr['x'].mean():.4g}  x_std={arr['x'].std():.4g}")

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = load_pickle("datasets/" + dataset_name + "/min.pkl")
        _max = load_pickle("datasets/" + dataset_name + "/max.pkl")

        for split in ('train', 'val', 'test'):
            yt = np.squeeze(np.transpose(data_dict['y_' + split], axes=[0, 2, 1, 3]), axis=-1)
            yt_normed = max_min_normalization(yt, _max[:, :, 0, :], _min[:, :, 0, :])
            data_dict['y_' + split] = np.transpose(yt_normed, axes=[0, 2, 1])

        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'],   data_dict['y_val'],   valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'],  data_dict['y_test'],  test_batch_size)
        data_dict['scaler'] = re_max_min_normalization
    else:
        scaler = StandardScaler(data_1d=data_dict['x_train'][..., 0])
        for mode in ['train', 'val', 'test']:
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])
        data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']   = DataLoader(data_dict['x_val'],   data_dict['y_val'],   valid_batch_size)
        data_dict['test_loader']  = DataLoader(data_dict['x_test'],  data_dict['y_test'],  test_batch_size)
        data_dict['scaler'] = scaler

    return data_dict


# ─── v9: dispatch-dict load_adj ───

_ADJ_DISPATCH = {
    "scalap":          lambda A: [calculate_scaled_laplacian(A).astype(np.float32).todense()],
    "normlap":         lambda A: [calculate_symmetric_normalized_laplacian(A).astype(np.float32).todense()],
    "symnadj":         lambda A: [symmetric_message_passing_adj(A).astype(np.float32).todense()],
    "transition":      lambda A: [transition_matrix(A).T],
    "doubletransition":lambda A: [transition_matrix(A).T, transition_matrix(A.T).T],
    "identity":        lambda A: [np.diag(np.ones(A.shape[0])).astype(np.float32)],
    "original":        lambda A: A,
    "spectral":        lambda A: [_spectral_low_pass(A)],    # v9 新增
}


def load_adj(file_path, adj_type):
    try:
        _, _, adj_mx = load_pickle(file_path)
    except (ValueError, TypeError):
        adj_mx = load_pickle(file_path)

    builder = _ADJ_DISPATCH.get(adj_type)
    if builder is None:
        raise ValueError(f"unknown adj_type '{adj_type}', "
                         f"choose from {list(_ADJ_DISPATCH.keys())}")
    adj = builder(adj_mx)
    _dbg(_TAG, f"load_adj  type={adj_type}  n_matrices={len(adj) if isinstance(adj, list) else 'raw'}")
    return adj, adj_mx
