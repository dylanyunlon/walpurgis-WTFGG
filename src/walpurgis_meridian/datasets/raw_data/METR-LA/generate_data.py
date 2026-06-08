"""Generate train/val/test splits for traffic data."""
import numpy as np
import os, argparse

def gen_seq2seq_io(data, x_off, y_off, add_tod=True, add_dow=True):
    ns, nn = data.shape[0], data.shape[1]
    dl = [data]
    if add_tod:
        ti = (np.arange(ns) % 288) / 288
        dl.append(np.tile(ti, [1, nn, 1]).transpose((2, 1, 0)))
    if add_dow:
        dw = np.zeros((ns, nn, 7))
        dw[np.arange(ns), :, (np.arange(ns) // 288) % 7] = 1
        dl.append(dw)
    data = np.concatenate(dl, axis=-1)
    x, y = [], []
    for t in range(abs(min(x_off)), ns - abs(max(y_off))):
        x.append(data[t + x_off])
        y.append(data[t + y_off])
    return np.stack(x), np.stack(y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./")
    p.add_argument("--traffic_df_filename", required=True)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    a = p.parse_args()
    data = np.load(a.traffic_df_filename)["data"]
    if data.ndim == 3: data = data[:, :, 0:1]
    x, y = gen_seq2seq_io(data, np.sort(np.arange(-(a.window-1), 1)), np.sort(np.arange(1, a.horizon+1)))
    n = x.shape[0]; nt = round(n*0.2); nr = round(n*0.7); nv = n - nt - nr
    for cat, sx, sy in [("train",x[:nr],y[:nr]),("val",x[nr:nr+nv],y[nr:nr+nv]),("test",x[-nt:],y[-nt:])]:
        print(f"{cat}: x={sx.shape} y={sy.shape}")
        np.savez_compressed(os.path.join(a.output_dir, cat+".npz"), x=sx, y=sy)

if __name__ == "__main__": main()

