"""
generate_training_data.py (PEMS04) — v9 port
Algo delta:
  1. MinMaxnormalization → RobustMinMax:
     先把数据 clip 到 [p1, p99] percentile, 再做 MinMax.
     交通流量有极端尖峰 (事故/节假日), 原始 MinMax 导致正常值被压缩到很窄的范围.
     clip 后尖峰被截断, 正常流量的归一化分辨率更高.
  2. 时间编码: cyclic sin/cos (同 METR-LA)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os, pickle, math
import numpy as np

num_feat = 1


def RobustMinMaxNormalization(train, val, test, lo_pct=1, hi_pct=99):
    """
    v9: percentile-clipped MinMax.
    先在 train 上计算 p1/p99 → clip → 再 MinMax 到 [-1, 1].
    """
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]
    # percentile thresholds from training set only
    _lo = np.percentile(train, lo_pct, axis=(0, 1, 3), keepdims=True)
    _hi = np.percentile(train, hi_pct, axis=(0, 1, 3), keepdims=True)
    print(f"v9 RobustMinMax  p{lo_pct}={_lo.mean():.4f}  p{hi_pct}={_hi.mean():.4f}")

    def normalize(x):
        x = np.clip(x, _lo, _hi)
        x = 1.0 * (x - _lo) / (_hi - _lo + 1e-8)
        x = 2.0 * x - 1.0
        return x

    return {'_max': _hi, '_min': _lo}, normalize(train), normalize(val), normalize(test)


def generate_graph_seq2seq_io_data(
    data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]

    if add_time_in_day:
        # v9: cyclic sin/cos
        frac = np.array([i % 288 / 288.0 for i in range(num_samples)])
        sin_t = np.tile(np.sin(2 * np.pi * frac), [1, num_nodes, 1]).transpose((2, 1, 0))
        cos_t = np.tile(np.cos(2 * np.pi * frac), [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(sin_t)
        feature_list.append(cos_t)

    if add_day_in_week:
        dow = np.array([(i // 288) % 7 for i in range(num_samples)], dtype=float)
        sin_d = np.tile(np.sin(2 * np.pi * dow / 7), [1, num_nodes, 1]).transpose((2, 1, 0))
        cos_d = np.tile(np.cos(2 * np.pi * dow / 7), [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(sin_d)
        feature_list.append(cos_d)

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
    x_offsets = np.sort(np.concatenate((np.arange(-(args.seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (args.seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets,
                                           add_time_in_day=True, add_day_in_week=args.dow)
    print("x shape:", x.shape, ", y shape:", y.shape)

    num_samples = x.shape[0]
    num_test  = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val   = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val,   y_val   = x[num_train:num_train+num_val], y[num_train:num_train+num_val][..., 0:1]
    x_test,  y_test  = x[-num_test:], y[-num_test:][..., 0:1]

    # v9: RobustMinMax normalization
    x_tr_feat = np.transpose(x_train[:, :, :, :num_feat], [0, 2, 3, 1])
    x_va_feat = np.transpose(x_val  [:, :, :, :num_feat], [0, 2, 3, 1])
    x_te_feat = np.transpose(x_test [:, :, :, :num_feat], [0, 2, 3, 1])

    stat, x_tr_feat, x_va_feat, x_te_feat = RobustMinMaxNormalization(
        x_tr_feat, x_va_feat, x_te_feat)

    x_train = np.concatenate([np.transpose(x_tr_feat, [0, 3, 1, 2]), x_train[:, :, :, num_feat:]], axis=-1)
    x_val   = np.concatenate([np.transpose(x_va_feat, [0, 3, 1, 2]), x_val  [:, :, :, num_feat:]], axis=-1)
    x_test  = np.concatenate([np.transpose(x_te_feat, [0, 3, 1, 2]), x_test [:, :, :, num_feat:]], axis=-1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(args.output_dir, f"{cat}.npz"),
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

    pickle.dump(stat['_max'], open("datasets/PEMS04/max.pkl", 'wb'))
    pickle.dump(stat['_min'], open("datasets/PEMS04/min.pkl", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/PEMS04")
    parser.add_argument("--traffic_df_filename", type=str, default="datasets/raw_data/PEMS04/PEMS04.npz")
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--dow", type=bool, default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)
