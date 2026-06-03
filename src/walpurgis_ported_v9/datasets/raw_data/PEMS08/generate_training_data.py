"""
generate_training_data.py (PEMS08) — v9 port
Algo delta (same as PEMS04):
  1. RobustMinMax (p1/p99 clip) 替代原始 MinMax
  2. cyclic sin/cos 时间编码
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os, pickle
import numpy as np

num_feat = 1


def RobustMinMaxNormalization(train, val, test, lo_pct=1, hi_pct=99):
    _lo = np.percentile(train, lo_pct, axis=(0, 1, 3), keepdims=True)
    _hi = np.percentile(train, hi_pct, axis=(0, 1, 3), keepdims=True)
    def normalize(x):
        x = np.clip(x, _lo, _hi)
        x = 1.0 * (x - _lo) / (_hi - _lo + 1e-8)
        return 2.0 * x - 1.0
    return {'_max': _hi, '_min': _lo}, normalize(train), normalize(val), normalize(test)


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets,
                                    add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]
    if add_time_in_day:
        frac = np.array([i % 288 / 288.0 for i in range(num_samples)])
        feature_list.append(np.tile(np.sin(2*np.pi*frac), [1,num_nodes,1]).transpose((2,1,0)))
        feature_list.append(np.tile(np.cos(2*np.pi*frac), [1,num_nodes,1]).transpose((2,1,0)))
    if add_day_in_week:
        dow = np.array([(i//288)%7 for i in range(num_samples)], dtype=float)
        feature_list.append(np.tile(np.sin(2*np.pi*dow/7), [1,num_nodes,1]).transpose((2,1,0)))
        feature_list.append(np.tile(np.cos(2*np.pi*dow/7), [1,num_nodes,1]).transpose((2,1,0)))
    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return np.stack(x, axis=0), np.stack(y, axis=0)


def generate_train_val_test(args):
    data = np.load(args.traffic_df_filename)['data']
    x_offsets = np.sort(np.concatenate((np.arange(-(args.seq_length_x-1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.seq_length_y+1), 1))
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets,
                                           add_time_in_day=True, add_day_in_week=args.dow)
    n = x.shape[0]
    n_test  = round(n * 0.2)
    n_train = round(n * 0.6)
    n_val   = n - n_test - n_train

    x_train, y_train = x[:n_train], y[:n_train][..., 0:1]
    x_val,   y_val   = x[n_train:n_train+n_val], y[n_train:n_train+n_val][..., 0:1]
    x_test,  y_test  = x[-n_test:], y[-n_test:][..., 0:1]

    x_tr_f = np.transpose(x_train[:,:,:,:num_feat], [0,2,3,1])
    x_va_f = np.transpose(x_val  [:,:,:,:num_feat], [0,2,3,1])
    x_te_f = np.transpose(x_test [:,:,:,:num_feat], [0,2,3,1])
    stat, x_tr_f, x_va_f, x_te_f = RobustMinMaxNormalization(x_tr_f, x_va_f, x_te_f)

    x_train = np.concatenate([np.transpose(x_tr_f,[0,3,1,2]), x_train[:,:,:,num_feat:]], axis=-1)
    x_val   = np.concatenate([np.transpose(x_va_f,[0,3,1,2]), x_val  [:,:,:,num_feat:]], axis=-1)
    x_test  = np.concatenate([np.transpose(x_te_f,[0,3,1,2]), x_test [:,:,:,num_feat:]], axis=-1)

    for cat in ["train","val","test"]:
        _x, _y = locals()["x_"+cat], locals()["y_"+cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(args.output_dir, f"{cat}.npz"),
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(list(x_offsets.shape)+[1]),
                            y_offsets=y_offsets.reshape(list(y_offsets.shape)+[1]))
    pickle.dump(stat['_max'], open("datasets/PEMS08/max.pkl",'wb'))
    pickle.dump(stat['_min'], open("datasets/PEMS08/min.pkl",'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/PEMS08")
    parser.add_argument("--traffic_df_filename", type=str, default="datasets/raw_data/PEMS08/PEMS08.npz")
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
