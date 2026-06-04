#!/usr/bin/env python3
"""
生成合成交通流数据, 格式与 METR-LA 完全对齐.

真实 METR-LA: 207 sensor, 12 step in, 12 step out, 2 feat (speed, time_of_day)
合成版: 20 sensor (跑得快), 相同 shape 约定.

同时生成 adj_mx pkl (距离矩阵).
"""
import os
import pickle
import numpy as np

np.random.seed(42)

NUM_NODES = 20
SEQ_LEN = 12
HORIZON = 12
NUM_FEAT = 3      # [speed, time_of_day_slot, day_of_week]
NUM_SAMPLES_TRAIN = 400
NUM_SAMPLES_VAL = 100
NUM_SAMPLES_TEST = 100

OUT_DIR = "walpurgis/datasets/METR-LA"
ADJ_DIR = "walpurgis/datasets/sensor_graph"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ADJ_DIR, exist_ok=True)

print(f"[synth] Generating synthetic METR-LA data")
print(f"  nodes={NUM_NODES}, seq={SEQ_LEN}, horizon={HORIZON}")
print(f"  train={NUM_SAMPLES_TRAIN}, val={NUM_SAMPLES_VAL}, test={NUM_SAMPLES_TEST}")

# ---- Generate traffic speed-like data ----
# 合成策略: 基础速度 ~60 mph + 周期(day) + 空间相关性 + 噪声
# 让每个sensor有不同的baseline, 加入合理的时空模式

sensor_baselines = 40 + 30 * np.random.rand(NUM_NODES)  # 40~70 mph

def generate_split(n_samples, split_name):
    # x: (N, T_in, nodes, feat)  y: (N, T_out, nodes, feat)
    x_all = np.zeros((n_samples, SEQ_LEN, NUM_NODES, NUM_FEAT), dtype=np.float32)
    y_all = np.zeros((n_samples, HORIZON, NUM_NODES, NUM_FEAT), dtype=np.float32)

    for i in range(n_samples):
        t_start = np.random.randint(0, 288 - SEQ_LEN - HORIZON)  # 5-min slots/day
        for t_offset in range(SEQ_LEN + HORIZON):
            t = t_start + t_offset
            tod = t / 288.0  # time of day normalized
            # 日内周期: 早晚高峰减速
            rush_hour = 0.7 + 0.3 * np.cos(2 * np.pi * (tod - 0.35))
            speed = sensor_baselines * rush_hour + np.random.randn(NUM_NODES) * 5
            speed = np.clip(speed, 0, 100)

            if t_offset < SEQ_LEN:
                x_all[i, t_offset, :, 0] = speed
                x_all[i, t_offset, :, 1] = tod      # 会被 *288 -> LongTensor索引
                x_all[i, t_offset, :, 2] = np.random.randint(0, 7)  # day_of_week
            else:
                y_all[i, t_offset - SEQ_LEN, :, 0] = speed
                y_all[i, t_offset - SEQ_LEN, :, 1] = tod
                y_all[i, t_offset - SEQ_LEN, :, 2] = np.random.randint(0, 7)

    path = os.path.join(OUT_DIR, f"{split_name}.npz")
    np.savez_compressed(path, x=x_all, y=y_all)
    print(f"  [{split_name}] x={x_all.shape}, y={y_all.shape} -> {path}")
    print(f"    speed range: [{x_all[...,0].min():.1f}, {x_all[...,0].max():.1f}]")
    return x_all, y_all

generate_split(NUM_SAMPLES_TRAIN, "train")
generate_split(NUM_SAMPLES_VAL, "val")
generate_split(NUM_SAMPLES_TEST, "test")

# ---- Generate adjacency matrix ----
# 模拟sensor之间的距离: 随机坐标, 欧式距离
coords = np.random.rand(NUM_NODES, 2) * 10  # 10km x 10km
dist_mx = np.zeros((NUM_NODES, NUM_NODES))
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        dist_mx[i, j] = np.linalg.norm(coords[i] - coords[j])

# 转成权重矩阵: exp(-dist^2 / sigma^2), sigma=中位数
sigma = np.median(dist_mx[dist_mx > 0])
adj_mx = np.exp(-dist_mx ** 2 / sigma ** 2)
np.fill_diagonal(adj_mx, 0)

# 格式: (sensor_ids, sensor_id_to_ind, adj_mx)
sensor_ids = [f"sensor_{i}" for i in range(NUM_NODES)]
sensor_id_to_ind = {s: i for i, s in enumerate(sensor_ids)}

adj_path = os.path.join(ADJ_DIR, "adj_mx_la.pkl")
with open(adj_path, 'wb') as f:
    pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f)
print(f"\n  adj_mx: ({NUM_NODES},{NUM_NODES}), sigma={sigma:.2f} -> {adj_path}")
print(f"  adj range: [{adj_mx.min():.4f}, {adj_mx.max():.4f}]")
print(f"\n[synth] Done.")
