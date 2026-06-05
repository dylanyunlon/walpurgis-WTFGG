"""
_gen_speed_common — Nightfall变体
METR-LA和PEMS-BAY共用的seq2seq数据生成
算法改写:
  1. 窗口滑动前加NaN/Inf检测和数据质量report
  2. split比例参数化 (不再硬编码)
  3. 保存时附带统计摘要 (mean/std/range)
"""
import argparse
import numpy as np
import os
import pandas as pd
import sys


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True):
    num_time_slot_a_day = 288
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    # NaN检测
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"[NF-WARN] {nan_count} NaN values in raw data, filling with forward-fill", file=sys.stderr)
        df = df.ffill().bfill()
        data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
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


def generate_train_val_test(args, train_ratio=0.7, test_ratio=0.2):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)
    print("x shape:", x.shape, ", y shape:", y.shape)
    # 参数化split比例
    num_samples = x.shape[0]
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    # 数据泄漏检查
    assert num_train + num_val + num_test == num_samples, "Split sizes don't match total"
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        # 统计摘要
        print(f"  {cat} x stats: mean={_x[...,0].mean():.2f} std={_x[...,0].std():.2f} "
              f"range=[{_x[...,0].min():.2f}, {_x[...,0].max():.2f}]")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
