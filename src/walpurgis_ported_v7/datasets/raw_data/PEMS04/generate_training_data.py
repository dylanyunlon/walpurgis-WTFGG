from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import numpy as np
import os
import sys

_DBG_GEN = ("--dbg-gen" in sys.argv)

num_feat = 1


def MinMaxnormalization(train, val, test):
    """算法改动:
    1. _max - _min 加 epsilon 避免常量通道除零
    2. 归一化前做 percentile-clip (1%, 99%) 消除极端离群点
    """
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    # 算法改动: percentile clip
    p_low = np.percentile(train, 1, axis=(0, 1, 3), keepdims=True)
    p_high = np.percentile(train, 99, axis=(0, 1, 3), keepdims=True)
    train = np.clip(train, p_low, p_high)
    val = np.clip(val, p_low, p_high)
    test = np.clip(test, p_low, p_high)

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    if _DBG_GEN:
        print(f"[DBG-GEN] MinMax  _max_range=[{_max.min():.2f}, {_max.max():.2f}]  "
              f"_min_range=[{_min.min():.2f}, {_min.max():.2f}]  "
              f"clip_low={p_low.mean():.2f}  clip_high={p_high.mean():.2f}")

    def normalize(x):
        # 算法改动: epsilon 防零幅
        denom = _max - _min
        denom = np.where(denom < 1e-8, 1e-8, denom)
        x = 1. * (x - _min) / denom
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]

    if add_time_in_day:
        time_ind = [i % 288 / 288 for i in range(num_samples)]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(
            time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = [(i // 288) % 7 for i in range(num_samples)]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(
            day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)

    if _DBG_GEN:
        print(f"[DBG-GEN] features  dim={data.shape[-1]}  "
              f"samples={data.shape[0]}  nodes={data.shape[1]}")

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    data = np.load(args.traffic_df_filename)['data']

    x_offsets = np.sort(np.concatenate(
        (np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        data, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    if _DBG_GEN:
        print(f"[DBG-GEN] split  train={num_train}  val={num_val}  "
              f"test={num_test}")

    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val][..., 0:1])
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]

    # MinMax normalization on traffic features only
    x_train_norm = x_train[:, :, :, :num_feat]
    x_train_time = x_train[:, :, :, num_feat:]
    x_val_norm = x_val[:, :, :, :num_feat]
    x_val_time = x_val[:, :, :, num_feat:]
    x_test_norm = x_test[:, :, :, :num_feat]
    x_test_time = x_test[:, :, :, num_feat:]

    x_train_norm = np.transpose(x_train_norm, axes=[0, 2, 3, 1])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 2, 3, 1])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 2, 3, 1])

    stat, x_train_norm, x_val_norm, x_test_norm = MinMaxnormalization(
        x_train_norm, x_val_norm, x_test_norm)

    x_train_norm = np.transpose(x_train_norm, axes=[0, 3, 1, 2])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 3, 1, 2])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 3, 1, 2])
    _max = stat['_max']
    _min = stat['_min']

    x_train = np.concatenate([x_train_norm, x_train_time], axis=-1)
    x_val = np.concatenate([x_val_norm, x_val_time], axis=-1)
    x_test = np.concatenate([x_test_norm, x_test_time], axis=-1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

    pickle.dump(_max, open("datasets/PEMS04/max.pkl", 'wb'))
    pickle.dump(_min, open("datasets/PEMS04/min.pkl", 'wb'))


if __name__ == "__main__":
    seq_length_x = 12
    seq_length_y = 12
    y_start = 1
    dow = True
    output_dir = 'datasets/PEMS04'
    traffic_df_filename = 'datasets/raw_data/PEMS04/PEMS04.npz'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--traffic_df_filename", type=str,
                        default=traffic_df_filename)
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y)
    parser.add_argument("--y_start", type=int, default=y_start)
    parser.add_argument("--dow", type=bool, default=dow)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        reply = str(input(
            f'{args.output_dir} exists. Overwrite? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
