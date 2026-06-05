"""
_gen_flow_common — Nightfall变体
PEMS04和PEMS08共用的流量数据生成
算法改写:
  1. MinMaxnormalization加eps防护防除零
  2. 窗口生成前加异常值检测 (z-score>10的极端值)
  3. split比例参数化
"""
import argparse
import pickle
import numpy as np
import os
import sys

num_feat = 1


def MinMaxnormalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)
    eps = 1e-8
    def normalize(x):
        x = 1. * (x - _min) / (_max - _min + eps)
        x = 2. * x - 1.
        return x
    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)
    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True):
    num_samples, num_nodes, _ = data.shape
    # 异常值检测
    raw = data[..., 0]
    mean_val, std_val = raw.mean(), raw.std()
    extreme = np.abs(raw - mean_val) > 10 * std_val
    if extreme.sum() > 0:
        print(f"[NF-WARN] {extreme.sum()} extreme values (z>10) detected", file=sys.stderr)
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
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args, dataset_name, train_ratio=0.6, test_ratio=0.2):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    data = np.load(args.traffic_df_filename)['data']
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(
        data, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)
    print("x shape:", x.shape, ", y shape:", y.shape)
    num_samples = x.shape[0]
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val][..., 0:1]
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]
    # MinMax归一化
    x_train_norm = np.transpose(x_train[:, :, :, :num_feat], axes=[0, 2, 3, 1])
    x_val_norm = np.transpose(x_val[:, :, :, :num_feat], axes=[0, 2, 3, 1])
    x_test_norm = np.transpose(x_test[:, :, :, :num_feat], axes=[0, 2, 3, 1])
    stat, x_train_norm, x_val_norm, x_test_norm = MinMaxnormalization(
        x_train_norm, x_val_norm, x_test_norm)
    x_train_norm = np.transpose(x_train_norm, axes=[0, 3, 1, 2])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 3, 1, 2])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 3, 1, 2])
    x_train = np.concatenate([x_train_norm, x_train[:, :, :, num_feat:]], axis=-1)
    x_val = np.concatenate([x_val_norm, x_val[:, :, :, num_feat:]], axis=-1)
    x_test = np.concatenate([x_test_norm, x_test[:, :, :, num_feat:]], axis=-1)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
    pickle.dump(stat['_max'], open(f"datasets/{dataset_name}/max.pkl", 'wb'))
    pickle.dump(stat['_min'], open(f"datasets/{dataset_name}/min.pkl", 'wb'))
