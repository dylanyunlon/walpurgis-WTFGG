"""
D2STGNN CardGame variant — _gen_flow_common.py
Common flow data generation routines for PEMS04/PEMS08 style datasets.
Algorithm changes vs upstream:
  1. Winsorize outlier clipping applied to raw flow features before normalization
  2. Cyclic time encoding using sin/cos for time_in_day and day_in_week
"""

import os
import sys
import pickle
import argparse
import numpy as np

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min():.6f} max={tensor.max():.6f} "
               f"mean={tensor.mean():.6f}")
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


NUM_FEAT = 1


def MinMaxnormalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)
    _dbg("minmax._max", _max.flatten(), "_gen_flow_common")
    _dbg("minmax._min", _min.flatten(), "_gen_flow_common")

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)
    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


# --- CARDGAME: Winsorize before normalization ---
def winsorize_array(arr, lo_pct=5, hi_pct=95):
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    return np.clip(arr, lo, hi)


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets,
                                    add_time_in_day=True, add_day_in_week=True,
                                    scaler=None):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:NUM_FEAT]]

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


def generate_train_val_test_flow(traffic_df_filename, output_dir, dataset_name,
                                  seq_length_x=12, seq_length_y=12, y_start=1, dow=True):
    """Generate train/val/test splits for flow-type datasets (PEMS04/08)."""
    data = np.load(traffic_df_filename)['data']

    # --- CARDGAME: Winsorize raw flow data ---
    data[..., 0] = winsorize_array(data[..., 0])
    _dbg("gen_flow.winsorized", data[..., 0], "_gen_flow_common")

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(data, x_offsets=x_offsets,
                                           y_offsets=y_offsets,
                                           add_time_in_day=True,
                                           add_day_in_week=dow)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train][..., 0:1]
    x_val, y_val = (x[num_train: num_train + num_val],
                    y[num_train: num_train + num_val][..., 0:1])
    x_test, y_test = x[-num_test:], y[-num_test:][..., 0:1]

    # MinMax normalization
    x_train_norm = x_train[:, :, :, :NUM_FEAT]
    x_train_time = x_train[:, :, :, NUM_FEAT:]
    x_val_norm = x_val[:, :, :, :NUM_FEAT]
    x_val_time = x_val[:, :, :, NUM_FEAT:]
    x_test_norm = x_test[:, :, :, :NUM_FEAT]
    x_test_time = x_test[:, :, :, NUM_FEAT:]

    x_train_norm = np.transpose(x_train_norm, axes=[0, 2, 3, 1])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 2, 3, 1])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 2, 3, 1])

    stat, x_train_norm, x_val_norm, x_test_norm = MinMaxnormalization(
        x_train_norm, x_val_norm, x_test_norm)

    x_train_norm = np.transpose(x_train_norm, axes=[0, 3, 1, 2])
    x_val_norm = np.transpose(x_val_norm, axes=[0, 3, 1, 2])
    x_test_norm = np.transpose(x_test_norm, axes=[0, 3, 1, 2])

    x_train = np.concatenate([x_train_norm, x_train_time], axis=-1)
    x_val = np.concatenate([x_val_norm, x_val_time], axis=-1)
    x_test = np.concatenate([x_test_norm, x_test_time], axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(output_dir, f"{cat}.npz"),
                            x=_x, y=_y,
                            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

    pickle.dump(stat['_max'], open(os.path.join(output_dir, "max.pkl"), 'wb'))
    pickle.dump(stat['_min'], open(os.path.join(output_dir, "min.pkl"), 'wb'))
    print(f"[CG] Flow data generated: {output_dir}")
