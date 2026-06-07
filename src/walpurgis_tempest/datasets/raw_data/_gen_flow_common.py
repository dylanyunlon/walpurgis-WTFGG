"""Common flow data generation (PEMS04, PEMS08)."""
import os, sys, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def generate_flow_data(data_path, output_dir, seq_x=12, seq_y=12):
    print(f"[TEM] Generate flow data: {data_path} -> {output_dir}")
    data = np.load(data_path)['data'][:, :, 0:1]
    T, N, F = data.shape
    _max = data.max(axis=0, keepdims=True); _min = data.min(axis=0, keepdims=True)
    t_day = (np.arange(T) % 288 / 288.0).reshape(-1, 1, 1) * np.ones((1, N, 1))
    d_week = (np.arange(T) // 288 % 7).reshape(-1, 1, 1) * np.ones((1, N, 1))
    data = np.concatenate([data, t_day, d_week], axis=-1)
    x_offsets = np.arange(-(seq_x - 1), 1, 1); y_offsets = np.arange(1, seq_y + 1, 1)
    xs, ys = [], []
    for t in range(seq_x - 1, T - seq_y):
        xs.append(data[t + x_offsets, ...]); ys.append(data[t + y_offsets, ..., :1])
    x = np.stack(xs); y = np.stack(ys)
    n = x.shape[0]; n_train = int(n * 0.7); n_test = int(n * 0.2); n_val = n - n_train - n_test
    os.makedirs(output_dir, exist_ok=True)
    for name, sx, ex in [('train', 0, n_train), ('val', n_train, n_train+n_val), ('test', n-n_test, n)]:
        np.savez_compressed(os.path.join(output_dir, f'{name}.npz'), x=x[sx:ex], y=y[sx:ex])
        print(f"  {name}: x={x[sx:ex].shape}")
    pickle.dump(_max, open(os.path.join(output_dir, 'max.pkl'), 'wb'))
    pickle.dump(_min, open(os.path.join(output_dir, 'min.pkl'), 'wb'))
