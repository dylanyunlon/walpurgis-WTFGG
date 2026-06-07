import numpy as np
import os
import sys

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    print(f"[EQX:synth:{tag}] {val}", file=sys.stderr)


def generate_spatiotemporal_data(num_nodes=20, seq_len=2000, num_features=3, seed=42):
    np.random.seed(seed)
    t = np.arange(seq_len).astype(float)
    data = np.zeros((seq_len, num_nodes, num_features), dtype=np.float32)
    for n in range(num_nodes):
        freq = 0.01 + 0.005 * n
        phase = np.random.uniform(0, 2 * np.pi)
        base = 50.0 + 10 * np.sin(2 * np.pi * freq * t + phase)
        base += 5 * np.sin(2 * np.pi * (freq * 3) * t + phase / 2)
        noise = np.random.randn(seq_len) * 2.0
        data[:, n, 0] = base + noise
        data[:, n, 1] = (t % 288) / 288.0
        data[:, n, 2] = (t % (288 * 7)) / (288 * 7)
    _edbg("generated", f"shape={data.shape} mean={data[:,:,0].mean():.2f} std={data[:,:,0].std():.2f}")
    return data


def create_sequences(data, seq_x=12, seq_y=12):
    x_list, y_list = [], []
    total = len(data) - seq_x - seq_y + 1
    for i in range(total):
        x_list.append(data[i:i + seq_x])
        y_list.append(data[i + seq_x:i + seq_x + seq_y])
    return np.array(x_list), np.array(y_list)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'raw_data', 'SYNTH')
    os.makedirs(output_dir, exist_ok=True)

    data = generate_spatiotemporal_data(num_nodes=20, seq_len=2000, num_features=3)
    x_all, y_all = create_sequences(data, seq_x=12, seq_y=12)

    n = len(x_all)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    splits = {
        'train': (x_all[:n_train], y_all[:n_train]),
        'val': (x_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]),
        'test': (x_all[n_train + n_val:], y_all[n_train + n_val:])
    }

    for name, (x, y) in splits.items():
        path = os.path.join(output_dir, f'{name}.npz')
        np.savez(path, x=x, y=y)
        _edbg(name, f"x={x.shape} y={y.shape}")
        print(f"Saved {name}: x={x.shape} y={y.shape} -> {path}")

    adj = np.random.rand(20, 20).astype(np.float32)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    adj = (adj > 0.7).astype(np.float32)
    adj_path = os.path.join(os.path.dirname(__file__), 'datasets', 'sensor_graph', 'adj_synth.npy')
    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    np.save(adj_path, adj)
    print(f"Saved adj: {adj.shape} nnz={int(adj.sum())} -> {adj_path}")


if __name__ == '__main__':
    main()
