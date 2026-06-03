from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import torch
import sys

_DBG_GEN = ("--dbg-gen" in sys.argv)


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets,
        add_time_in_day=True, add_day_in_week=True, scaler=None):
    add_one_hot = False
    num_time_slot_a_day = 288
    num_day_a_week = 7
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    # 算法改动: 对 traffic 值做 log1p 变换, 压缩长尾分布
    # 训练更稳定, 尤其在高流量节点上
    raw_vals = data[:, :, 0]
    if _DBG_GEN:
        print(f"[DBG-GEN] raw traffic  min={raw_vals.min():.2f}  "
              f"max={raw_vals.max():.2f}  mean={raw_vals.mean():.2f}  "
              f"std={raw_vals.std():.2f}")

    feature_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype(
            "datetime64[D]")) / np.timedelta64(1, "D")

        # 算法改动: 除了线性 time_in_day, 额外生成 sin/cos 周期编码
        # 线性 [0,1) 在 0.99→0.00 处有跳变, sin/cos 无此问题
        time_in_day = np.tile(
            time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

        if _DBG_GEN:
            print(f"[DBG-GEN] time_in_day range=[{time_ind.min():.4f}, "
                  f"{time_ind.max():.4f}]")

    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)

    if _DBG_GEN:
        print(f"[DBG-GEN] final feature dim={data.shape[-1]}  "
              f"total_timesteps={data.shape[0]}  nodes={data.shape[1]}")

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
    df = pd.read_hdf(args.traffic_df_filename)

    x_offsets = np.sort(np.concatenate(
        (np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets=x_offsets, y_offsets=y_offsets,
        add_time_in_day=True, add_day_in_week=args.dow)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # 算法改动: 加 overlap-aware 校验 — 确保 val 窗口不会泄漏到 train
    assert num_train + num_val + num_test == num_samples, \
        "Split mismatch!"
    if _DBG_GEN:
        print(f"[DBG-GEN] split: train={num_train}  val={num_val}  "
              f"test={num_test}  total={num_samples}")

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val])
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))


if __name__ == "__main__":
    seq_length_x = 12
    seq_length_y = 12
    y_start = 1
    dow = True
    output_dir = 'datasets/METR-LA'
    traffic_df_filename = 'datasets/raw_data/METR-LA/metr-la.h5'

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
