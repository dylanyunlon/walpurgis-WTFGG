from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import pickle
import numpy as np
import os

num_feat = 1

# 改动1: MinMaxNorm 加 epsilon 防零除
# upstream: 直接 (x - min) / (max - min), max=min 时爆炸
_MM_EPS = 1e-8


def MinMaxnormalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print(f'_max.shape: {_max.shape}, _min.shape: {_min.shape}')

    def normalize(x):
        denom = _max - _min
        # 改动1: epsilon 防零除
        denom = np.where(np.abs(denom) < _MM_EPS, _MM_EPS, denom)
        x = 1. * (x - _min) / denom
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    # 改动5: NaN 审计 — upstream 不检查
    for name, arr in [('train', train_norm), ('val', val_norm), ('test', test_norm)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"[walpurgis WARNING] {name} has {n_nan} NaN after norm, replacing with 0")
            arr[np.isnan(arr)] = 0.0

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True, stride=1):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]

    if add_time_in_day:
        time_ind = [i % 288 / 288 for i in range(num_samples)]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = [(i // 288) % 7 for i in range(num_samples)]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    # 改动2: stride 跳步
    for t in range(min_t, max_t, stride):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x = args.seq_length_x
    seq_length_y = args.seq_length_y
    data = np.load(args.traffic_df_filename)['data']

    x_offsets = np.sort(np.concatenate(
        (np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        data, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow,
        stride=args.stride)

    print(f"x shape: {x.shape}, y shape: {y.shape}")

    num_samples = x.shape[0]

    # 改动3: 按周对齐划分
    samples_per_week = 288 * 7
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_test = (num_test // samples_per_week) * samples_per_week or round(num_samples * 0.2)
    num_train = (num_train // samples_per_week) * samples_per_week or round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = (x[num_train:num_train + num_val],
                    y[num_train:num_train + num_val][..., 0:1])
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]

    # MinMax normalization on traffic features
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
        print(f"{cat}: x={_x.shape}, y={_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    pickle.dump(_max, open("datasets/PEMS04/max.pkl", 'wb'))
    pickle.dump(_min, open("datasets/PEMS04/min.pkl", 'wb'))

    # 改动4: stats JSON
    stats = {
        'total_samples': int(num_samples),
        'train': int(num_train), 'val': int(num_val), 'test': int(num_test),
        'num_nodes': int(x.shape[2]), 'feat_dim': int(x.shape[3]),
        'max_val': float(_max.max()), 'min_val': float(_min.min()),
        'stride': args.stride,
    }
    with open(os.path.join(args.output_dir, 'data_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[walpurgis] Stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='datasets/PEMS04')
    parser.add_argument("--traffic_df_filename", type=str,
                        default='datasets/raw_data/PEMS04/PEMS04.npz')
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(
            f'{args.output_dir} exists. Overwrite? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
