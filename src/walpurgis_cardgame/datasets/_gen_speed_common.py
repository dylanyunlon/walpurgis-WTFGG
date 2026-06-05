"""
_gen_speed_common — CardGame变体
METR-LA和PEMS-BAY共用的seq2seq数据生成

算法改写 (vs upstream):
  1. Winsorize离群值裁剪 (1st/99th percentile) 替代原始直接切分
  2. robust z-score归一化 (median/IQR) 用于数据质量报告
  3. SHA-256 checksum 保存到每个split的meta文件
"""
import argparse
import hashlib
import json
import numpy as np
import os
import pandas as pd
import sys

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG:
        return
    if hasattr(tensor, 'shape'):
        arr = np.asarray(tensor)
        msg = (f"[CG-DBG:{tag}] shape={list(arr.shape)} dtype={arr.dtype} "
               f"min={arr.min():.6f} max={arr.max():.6f} "
               f"mean={arr.mean():.6f} std={arr.std():.6f}")
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            msg += f" *** NaN={nan_count} ***"
        if inf_count > 0:
            msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def _winsorize(arr, lo_pct=1.0, hi_pct=99.0):
    """Winsorize极端值到指定百分位"""
    lo = np.nanpercentile(arr, lo_pct)
    hi = np.nanpercentile(arr, hi_pct)
    clipped = np.clip(arr, lo, hi)
    n_clipped = int((arr < lo).sum() + (arr > hi).sum())
    if _CG_DEBUG and n_clipped > 0:
        _dbg("winsorize", clipped, module="_gen_speed_common")
        print(f"[CG-DBG:winsorize] clipped {n_clipped} values to [{lo:.4f}, {hi:.4f}]",
              file=sys.stderr)
    return clipped


def _sha256_bytes(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


def _robust_zscore_report(arr, name):
    """用median/IQR做robust z-score统计"""
    flat = arr.flatten()
    median = np.nanmedian(flat)
    q25, q75 = np.nanpercentile(flat, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-8:
        iqr = 1.0
    z = (flat - median) / iqr
    n_outlier = int((np.abs(z) > 3.0).sum())
    report = {
        "name": name,
        "median": float(median),
        "iqr": float(iqr),
        "q25": float(q25),
        "q75": float(q75),
        "robust_outliers_gt3iqr": n_outlier,
        "total_elements": int(flat.size),
    }
    return report


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True):
    num_time_slot_a_day = 288
    num_samples, num_nodes = df.shape

    raw_values = df.values.copy()
    # Winsorize离群值裁剪
    winsorized = _winsorize(raw_values, lo_pct=1.0, hi_pct=99.0)
    data = np.expand_dims(winsorized, axis=-1)

    # NaN/Inf检测与修复
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"[CG-WARN] {nan_count} NaN, {inf_count} Inf in raw data → forward/back fill",
              file=sys.stderr)
        df_clean = pd.DataFrame(winsorized, index=df.index, columns=df.columns)
        df_clean = df_clean.ffill().bfill()
        data = np.expand_dims(df_clean.values, axis=-1)

    _dbg("speed_data_raw", data, module="_gen_speed_common")

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
    _dbg("speed_data_features", data, module="_gen_speed_common")

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

    num_samples = x.shape[0]
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    assert num_train + num_val + num_test == num_samples, "Split sizes don't match total"

    meta_all = {}
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)

        report = _robust_zscore_report(_x[..., 0], f"{cat}_x_speed")
        print(f"  {cat} robust stats: median={report['median']:.2f} "
              f"IQR={report['iqr']:.2f} outliers={report['robust_outliers_gt3iqr']}")

        out_path = os.path.join(args.output_dir, f"{cat}.npz")
        np.savez_compressed(
            out_path,
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

        with open(out_path, 'rb') as fh:
            checksum = _sha256_bytes(fh.read())
        report["sha256"] = checksum
        meta_all[cat] = report

    meta_path = os.path.join(args.output_dir, "split_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_all, f, indent=2)
    print(f"[CG] Meta saved: {meta_path}")
